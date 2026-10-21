"""Slice A1: toy codecs, exact known answers (design D4, D5)."""
import torch

from conftest import max_abs


def test_exact_roundtrip(get_codec):
    g = torch.Generator().manual_seed(0)
    x = 0.3 * torch.randn(1, 1, 8003, generator=g)  # not a multiple of h = 8
    for name in ("toy_causal", "toy_noncausal"):
        codec = get_codec(name)
        enc = codec.encode(x, 8000)
        assert enc.num_samples == 8003
        y = codec.decode(enc.z, enc.scale)
        assert y.shape == (1, 1, 8008)
        assert max_abs(y[..., :8003], x) <= 1e-5
        assert max_abs(y[..., 8003:], torch.zeros(1, 1, 5)) <= 1e-5


def test_scale_known(get_codec):
    codec = get_codec("toy_noncausal")
    x = torch.full((1, 1, 800), 0.5)
    enc = codec.encode(x, 8000)
    assert enc.scale is not None and enc.scale.shape == (1,)
    want = 0.5 + 1e-8
    assert abs(float(enc.scale[0]) - want) <= 1e-6 * want
    assert get_codec("toy_causal").encode(x, 8000).scale is None
