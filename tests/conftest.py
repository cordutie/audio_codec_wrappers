"""Shared fixtures for the audio_codec_wrappers tests (plan.md, Phase A).

Conventions (plan "Tests"):
- "<= eps" is the max absolute difference unless stated.
- Codec tests use samples/guitar.wav (mono, 44.1 kHz, 10.67 s) unless stated.
- GPU and weights are required, with no skips. A missing weight file is a failure
  whose message names scripts/fetch_weights.py.
"""
import os
import pathlib

# Preallocation off before anything can import jax in this process. The package
# itself must do the same (checked in a clean subprocess by test_env.py::test_prealloc_env).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import soundfile as sf
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
GUITAR = REPO / "samples" / "guitar.wav"

REAL_CODECS = ["encodec24k", "spectrostream", "encodec48k", "music2latent"]
TOY_CODECS = ["toy_causal", "toy_noncausal"]
ALL_CODECS = REAL_CODECS + TOY_CODECS
CAUSAL_CODECS = ["encodec24k", "spectrostream", "toy_causal"]
NONCAUSAL_CODECS = ["encodec48k", "music2latent", "toy_noncausal"]

DEVICE = "cuda"


def load_guitar(start_s: float = 0.0, end_s: float | None = None) -> tuple[torch.Tensor, int]:
    """Guitar sample as [1, 1, N] float32 at its native 44.1 kHz (read with soundfile, not the package)."""
    data, sr = sf.read(str(GUITAR), dtype="float32", always_2d=True)  # [N, C]
    x = torch.from_numpy(data.T.copy())  # [C, N]
    a = int(round(start_s * sr))
    b = x.shape[-1] if end_s is None else int(round(end_s * sr))
    return x[:, a:b].unsqueeze(0).contiguous(), sr


@pytest.fixture(scope="session")
def guitar():
    return load_guitar()


@pytest.fixture(scope="session")
def get_codec():
    """Returns load_codec(name, device=DEVICE); the registry caches per (name, device)."""
    def _get(name: str, device: str = DEVICE):
        from audio_codec_wrappers import load_codec
        return load_codec(name, device=device)
    return _get


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.as_tensor(a).detach().double().cpu()
    b = torch.as_tensor(b).detach().double().cpu()
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).abs().max()) if a.numel() else 0.0


def chunk_sizes(kind, total: int) -> list[int]:
    """Chunkings used by the streaming tests: all-1, all-7, random sizes 1..13 (seed 0)."""
    if kind == "random":
        g = torch.Generator().manual_seed(0)
        sizes, s = [], 0
        while s < total:
            n = int(torch.randint(1, 14, (1,), generator=g))
            n = min(n, total - s)
            sizes.append(n)
            s += n
        return sizes
    n = int(kind)
    return [min(n, total - i) for i in range(0, total, n)]


def stream(dec, z: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Push z [B, d, T] in chunks of the given sizes, then flush; returns cat of all outputs."""
    dec.reset()
    outs, t = [], 0
    for n in sizes:
        outs.append(dec.push(z[..., t:t + n]))
        t += n
    assert t == z.shape[-1]
    outs.append(dec.flush())
    return torch.cat(outs, dim=-1)
