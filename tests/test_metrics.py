"""Slice A1: metric implementations checked against known answers (design "Metrics", D18)."""
import math

import torch

SR = 48000


def _noise(seed: int, n: int = 2 * SR, c: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, n, generator=g)


def _click_train(period_s: float = 0.3, seconds: float = 3.0, sr: int = SR) -> torch.Tensor:
    x = torch.zeros(1, int(seconds * sr))
    x[0, :: int(period_s * sr)] = 1.0
    return x


def _delay(x: torch.Tensor, n: int) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[..., n:] = x[..., : x.shape[-1] - n]
    return y


def test_identity_zero():
    from audio_codec_wrappers.metrics import log_mel_distance
    x = _noise(0, c=2)
    assert abs(log_mel_distance(x, x, SR)) <= 1e-7


def test_gain_ln4():
    """Power mel: doubling the amplitude multiplies every band by 4, so M(x, 2x) = ln 4."""
    from audio_codec_wrappers.metrics import log_mel_distance
    x = _noise(0)
    m = log_mel_distance(x, 2 * x, SR)
    assert abs(m - math.log(4.0)) <= 0.01 * math.log(4.0)


def test_monotone():
    from audio_codec_wrappers.metrics import log_mel_distance
    x, n = _noise(0), _noise(1)
    nu = _noise(2)
    nu = nu * (x.pow(2).mean().sqrt() / nu.pow(2).mean().sqrt())  # RMS-matched noise
    a = log_mel_distance(x, x + 0.01 * n, SR)
    b = log_mel_distance(x, x + 0.1 * n, SR)
    c = log_mel_distance(x, nu, SR)
    assert a < b < c


def test_envelope_correlation_known():
    from audio_codec_wrappers.metrics import envelope_correlation, is_nonstationary
    clicks = _click_train()
    assert abs(envelope_correlation(clicks, clicks) - 1.0) <= 1e-6
    delayed = _delay(clicks, int(0.05 * SR))
    # Click envelopes 50 ms apart do not overlap (4 hops of 256 per click, 2400-sample delay):
    # the correlation is near -0.08, far below c(x, x) = 1.
    assert envelope_correlation(clicks, delayed) < 0.5
    assert is_nonstationary(clicks) is True
    assert is_nonstationary(_noise(3, n=3 * SR)) is False


def test_thresholds_rule():
    from audio_codec_wrappers.metrics import s1_thresholds
    tau = s1_thresholds({"a": [0.2, 0.3], "b": [0.7]})
    assert set(tau) == {"a", "b"}
    assert abs(tau["a"] - 0.45) <= 1e-12
    assert abs(tau["b"] - 0.9) <= 1e-12


def test_lag_known():
    from audio_codec_wrappers.metrics import envelope_lag
    x = _click_train()
    y = _delay(x, 1000)
    assert abs(envelope_lag(x, y, SR) - 1000) <= 256  # positive = y later than x
    assert abs(envelope_lag(x, x, SR)) <= 256
    assert abs(envelope_lag(y, x, SR) + 1000) <= 256
