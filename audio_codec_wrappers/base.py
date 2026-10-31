"""Common API for neural audio codecs (plan.md, Phase A Interfaces; design D3, D4, D5, D19).

Audio is float32 [B, C, N] in [-1, 1]; latents are float32 [B, d, T] (channel first).
"""
import abc
import math
from dataclasses import dataclass

import torch

from .audio import adapt_audio, pad_to_multiple


@dataclass(frozen=True)
class CodecInfo:
    name: str
    sample_rate: int                 # r
    channels: int                    # C
    hop_length: int                  # h, samples per latent frame
    latent_dim: int                  # d
    frame_quantum: int = 1           # q; encode/decode work on multiples of q frames
    causal: bool = False             # True iff streaming_decoder() is available
    decoder_lookahead_frames: int | None = None   # l; None iff not causal
    normalizes_input: bool = False   # True iff encode returns a scale
    stochastic_decoder: bool = False
    backend: str = "torch"           # "torch" | "jax"
    weights: str = ""                # weights id, e.g. "hf:<repo>@<sha256[:12]>"

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def num_frames(self, num_samples: int) -> int:
        """Frames of an input of num_samples samples (codec rate) after padding to a multiple of q*h."""
        block = self.frame_quantum * self.hop_length
        return self.frame_quantum * math.ceil(num_samples / block)


@dataclass
class Encoded:
    z: torch.Tensor                  # [B, d, T] float32 on the wrapper device, requires_grad=False
    scale: torch.Tensor | None       # [B] float32 iff info.normalizes_input
    num_samples: int                 # input length at codec rate, before padding


class StreamingDecoder(abc.ABC):
    """Stateful decoder. cat(push outputs..., flush()) has length T*h + L and its samples [L, L+T*h) equal
    the one-shot decode, for every chunking of z."""
    latency_samples: int             # L; output is delayed by L samples

    @abc.abstractmethod
    def push(self, z: torch.Tensor) -> torch.Tensor:
        """[B, d, n] -> [B, C, n*h]"""

    @abc.abstractmethod
    def flush(self) -> torch.Tensor:
        """Feeds ceil(L/h) zero frames; returns [B, C, L]."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Back to the initial (zero history) state."""


class WindowDecoder(StreamingDecoder):
    """Decodes a sliding window of the last window_frames latent frames from a fresh state (design D19).

    Only the instrument of the A6 window-versus-stream report; SCAPES never uses it.
    """

    def __init__(self, codec: "CodecWrapper", window_frames: int, batch_size: int = 1):
        self.codec = codec
        self.window_frames = window_frames
        self.batch_size = batch_size
        self.lookahead = codec.info.decoder_lookahead_frames
        self.latency_samples = self.lookahead * codec.info.hop_length
        self.reset()

    def reset(self) -> None:
        self.past = None  # [B, d, <= window_frames]

    def push(self, z: torch.Tensor) -> torch.Tensor:
        info = self.codec.info
        h, n = info.hop_length, z.shape[-1]
        past = z[..., :0] if self.past is None else self.past
        p = past.shape[-1]
        # decode the past and the new frames together from a fresh state
        y = self.codec.decode(torch.cat([past, z], dim=-1), None)
        # output frames [t - l, t + n - l) are local frames [p - l, p + n - l); zeros where there is no frame
        out = torch.zeros(y.shape[0], y.shape[1], n * h, dtype=y.dtype, device=y.device)
        lo = p - self.lookahead
        first = max(lo, 0)
        out[..., (first - lo) * h:] = y[..., first * h:(lo + n) * h]
        # keep the last window_frames frames
        joined = torch.cat([past, z], dim=-1)
        self.past = joined[..., max(joined.shape[-1] - self.window_frames, 0):] if self.window_frames > 0 else joined[..., :0]
        return out

    def flush(self) -> torch.Tensor:
        h = self.codec.info.hop_length
        k = math.ceil(self.latency_samples / h)
        zeros = torch.zeros(self.batch_size, self.codec.info.latent_dim, k, device=self.codec.device)
        out = self.push(zeros)
        return out[..., :self.latency_samples]


class CodecWrapper(abc.ABC):
    info: CodecInfo
    device: torch.device

    def prepare(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Accepts [N], [C_in, N] or [B, C_in, N]; resamples to info.sample_rate, maps channels;
        returns [B, C, N'] float32 on device."""
        audio = torch.as_tensor(audio).to(self.device)
        return adapt_audio(audio, sample_rate, self.info.sample_rate, self.info.channels)

    @abc.abstractmethod
    def _encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """audio [B, C, M] with M a multiple of q*h -> (z [B, d, M/h], scale or None)."""

    def encode(self, audio: torch.Tensor, sample_rate: int) -> Encoded:
        """prepare(), right-pad with zeros to a multiple of q*h, encode under torch.no_grad()."""
        x = self.prepare(audio, sample_rate)
        n = x.shape[-1]
        x = pad_to_multiple(x, self.info.frame_quantum * self.info.hop_length)
        with torch.no_grad():
            z, scale = self._encode(x)
        return Encoded(z=z.detach().float(), scale=None if scale is None else scale.detach().float(), num_samples=n)

    @abc.abstractmethod
    def decode(self, z: torch.Tensor, scale: torch.Tensor | None = None, seed: int | None = 0) -> torch.Tensor:
        """[B, d, T] -> [B, C, T*h], aligned with the encoder input (sample 0 <-> sample 0)."""

    def streaming_decoder(self, batch_size: int = 1) -> StreamingDecoder:
        raise NotImplementedError(f"{self.info.name} has no streaming decoder (not causal)")

    def window_decoder(self, window_frames: int, batch_size: int = 1) -> StreamingDecoder:
        if not self.info.causal:
            raise NotImplementedError(f"{self.info.name} has no window decoder (not causal)")
        return WindowDecoder(self, window_frames, batch_size)

    # Aliases (D3, R1-C10)
    encode_continuous = encode
    audio_to_latents = encode

    def latents_to_audio(self, z, scale=None, seed=0):
        return self.decode(z, scale, seed)
