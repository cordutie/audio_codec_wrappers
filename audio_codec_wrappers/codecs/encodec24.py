"""EnCodec 24 kHz (causal, mono) with a hand-written stateful streaming decoder (plan A3; C-4; design R1-C3, S1 (v)).

Latent = encoder output before the quantizer (128 x 75 Hz). `decode` is the zero-history causal decoder: the same decoder
with zero padding at the start instead of the official reflect padding, so that streaming equals it exactly.
`decode_official` is the unchanged official decoder (reflect padding at the start).
"""
import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CodecWrapper, StreamingDecoder
from ..registry import codec_info


@contextlib.contextmanager
def _full_precision():
    """TF32 off for the decoder. With TF32 on (the torch default for cuDNN convs), the rounding depends on the chunk shape
    and the streaming output differed from the one-shot decode by 2.5e-3 on the GPU (1.9e-6 with it off, slice A3)."""
    conv, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = conv, mm


def _conv_weight(conv: nn.Module) -> torch.Tensor:
    """Effective weight of a (weight-normed) conv, computed from g and v so that it follows .double() / .to()."""
    if hasattr(conv, "weight_g"):
        return torch._weight_norm(conv.weight_v, conv.weight_g, 0)
    return conv.weight


# Stages: one per decoder module, each with step(x [B, C, n]) -> [B, C', n'] and reset().
class _ConvStage:
    """Causal SConv1d, stride 1: the state is the last (k-1)*dilation input frames (zeros at the start)."""

    def __init__(self, module, batch_size):
        self.conv = module.conv.conv
        self.keep = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        assert self.conv.stride[0] == 1 and module.causal
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.state = None

    def step(self, x):
        if self.state is None:
            self.state = x.new_zeros(self.batch_size, x.shape[1], self.keep)
        x = torch.cat([self.state, x], dim=-1)
        self.state = x[..., x.shape[-1] - self.keep:] if self.keep > 0 else x[..., :0]
        return self.conv(x)


class _ConvTrStage:
    """Causal SConvTranspose1d (all trimming on the right): keeps the last k-s raw overlap-add samples, bias added once."""

    def __init__(self, module, batch_size):
        assert module.causal and module.trim_right_ratio == 1.0
        self.conv = module.convtr.convtr
        self.stride = self.conv.stride[0]
        self.tail_len = self.conv.kernel_size[0] - self.stride
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.tail = None

    def step(self, x):
        w = _conv_weight(self.conv)
        y = F.conv_transpose1d(x, w, None, stride=self.stride)  # [B, C', n*s + tail_len], no bias
        if self.tail is None:
            self.tail = y.new_zeros(self.batch_size, y.shape[1], self.tail_len)
        n_out = x.shape[-1] * self.stride
        head = y[..., :self.tail_len] + self.tail
        out = torch.cat([head, y[..., self.tail_len:n_out]], dim=-1) if n_out > self.tail_len else head[..., :n_out]
        # the tail beyond the emitted part carries over (n_out may be shorter than tail_len for tiny chunks)
        if n_out >= self.tail_len:
            self.tail = y[..., n_out:]
        else:
            rest = torch.cat([head[..., n_out:], y.new_zeros(self.batch_size, y.shape[1], n_out)], dim=-1)
            self.tail = rest + y[..., n_out:]
        if self.conv.bias is not None:
            out = out + self.conv.bias[None, :, None]
        return out


class _LstmStage:
    """SLSTM: (h, c) carried between pushes, plus the skip connection."""

    def __init__(self, module):
        self.lstm = module.lstm
        self.skip = module.skip
        self.reset()

    def reset(self):
        self.state = None

    def step(self, x):
        xt = x.permute(2, 0, 1)
        y, self.state = self.lstm(xt, self.state)
        if self.skip:
            y = y + xt
        return y.permute(1, 2, 0)


class _ModuleStage:
    """Stateless module (activation)."""

    def __init__(self, module):
        self.module = module

    def reset(self):
        pass

    def step(self, x):
        return self.module(x)


class _ResStage:
    """SEANetResnetBlock: shortcut(x) + block(x), each branch with its own state."""

    def __init__(self, module, batch_size):
        self.shortcut = _make_stage(module.shortcut, batch_size)
        self.block = [_make_stage(m, batch_size) for m in module.block]

    def reset(self):
        self.shortcut.reset()
        for s in self.block:
            s.reset()

    def step(self, x):
        y = x
        for s in self.block:
            y = s.step(y)
        return self.shortcut.step(x) + y


def _make_stage(module, batch_size):
    name = type(module).__name__
    if name == "SConv1d":
        return _ConvStage(module, batch_size)
    if name == "SConvTranspose1d":
        return _ConvTrStage(module, batch_size)
    if name == "SLSTM":
        return _LstmStage(module)
    if name == "SEANetResnetBlock":
        return _ResStage(module, batch_size)
    if name == "Identity" or isinstance(module, (nn.ELU, nn.Tanh)):
        return _ModuleStage(module)
    raise TypeError(f"no streaming stage for {name}")


class EncodecStreamingDecoder(StreamingDecoder):
    """Exact stateful decoder of the EnCodec 24k SEANetDecoder: no latency, output n*320 samples per n frames."""

    def __init__(self, decoder: nn.Module, batch_size: int = 1):
        self.decoder = decoder
        self.batch_size = batch_size
        self.latency_samples = 0
        self.stages = [_make_stage(m, batch_size) for m in decoder.model]
        self.dtype = next(decoder.parameters()).dtype
        self.device = next(decoder.parameters()).device

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def push(self, z: torch.Tensor) -> torch.Tensor:
        x = z.to(self.device, self.dtype)
        with torch.no_grad(), _full_precision():
            for s in self.stages:
                x = s.step(x)
        return x

    def flush(self) -> torch.Tensor:
        return torch.zeros(self.batch_size, self.decoder.channels, 0, device=self.device, dtype=self.dtype)


def _zero_history(decoder: nn.Module) -> nn.Module:
    """Copy of the decoder whose convolutions pad with zeros (constant) instead of reflect."""
    dec = copy.deepcopy(decoder)
    for m in dec.modules():
        if type(m).__name__ == "SConv1d":
            m.pad_mode = "constant"
    return dec


class Encodec24Codec(CodecWrapper):
    def __init__(self, device="cuda"):
        from encodec import EncodecModel
        self.info = codec_info("encodec24k")
        self.device = torch.device(device)
        self.model = EncodecModel.encodec_model_24khz().to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # weight_norm keeps a computed `weight` attribute that still carries a graph (and breaks deepcopy);
        # one forward pass without grad replaces it by a plain tensor
        with torch.no_grad():
            self.model.decoder(self.model.encoder(torch.zeros(1, 1, 3200, device=self.device)))
        self.zero_decoder = _zero_history(self.model.decoder).eval()

    def _encode(self, audio):
        # one item at a time with the default (TF32) settings: the result then does not depend on the batch, and equals the
        # official encoder run on the same item (the batch-dependence of TF32 rounding was 1e-2 in slice A3)
        z = torch.cat([self.model.encoder(a[None]) for a in audio], dim=0)
        return z, None

    def decode(self, z, scale=None, seed=0):
        z = z.to(self.device).float()
        if z.shape[-1] == 0:
            return z.new_zeros(z.shape[0], 1, 0)
        with torch.no_grad(), _full_precision():
            return self.zero_decoder(z)

    def decode_official(self, z, scale=None):
        """The official decoder unchanged (reflect padding at the start, so not equal to the streaming output)."""
        with torch.no_grad():
            return self.model.decoder(z.to(self.device).float())

    def streaming_decoder(self, batch_size: int = 1) -> StreamingDecoder:
        return EncodecStreamingDecoder(self.model.decoder, batch_size)
