"""Toy codecs for tests: frames of 8 samples mapped by a fixed orthogonal matrix (exact round trip)."""
import torch

from ..base import CodecInfo, CodecWrapper, StreamingDecoder

HOP, DIM = 8, 16


def _basis() -> torch.Tensor:
    """Q [16, 8] with orthonormal columns (QR of a seeded Gaussian matrix)."""
    g = torch.Generator().manual_seed(1234)
    q, _ = torch.linalg.qr(torch.randn(DIM, HOP, generator=g))
    return q


class ToyStreamingDecoder(StreamingDecoder):
    """Frame-wise decode: no memory, no latency."""

    def __init__(self, codec: "ToyCodec", batch_size: int = 1):
        self.codec = codec
        self.batch_size = batch_size
        self.latency_samples = 0

    def push(self, z: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(z, None)

    def flush(self) -> torch.Tensor:
        return torch.zeros(self.batch_size, 1, 0, device=self.codec.device)

    def reset(self) -> None:
        pass


class ToyCodec(CodecWrapper):
    def __init__(self, info: CodecInfo, device="cpu"):
        self.info = info
        self.device = torch.device(device)
        self.q = _basis().to(self.device)

    def _encode(self, audio):
        b = audio.shape[0]
        scale = None
        if self.info.normalizes_input:
            # sigma = RMS of the mono mixdown + 1e-8
            scale = audio.mean(dim=1).pow(2).mean(dim=-1).sqrt() + 1e-8
            audio = audio / scale[:, None, None]
        frames = audio.reshape(b, -1, HOP)  # [B, T, 8]
        return torch.einsum("dh,bth->bdt", self.q, frames), scale

    def decode(self, z, scale=None, seed=0):
        if self.info.normalizes_input and scale is None:
            raise ValueError(f"{self.info.name} needs the scale returned by encode")
        z = z.to(self.device)
        x = torch.einsum("dh,bdt->bth", self.q, z).reshape(z.shape[0], 1, -1)
        if self.info.normalizes_input:
            x = x * scale.to(self.device)[:, None, None]
        return x

    def streaming_decoder(self, batch_size: int = 1):
        if not self.info.causal:
            raise NotImplementedError(f"{self.info.name} has no streaming decoder (not causal)")
        return ToyStreamingDecoder(self, batch_size)
