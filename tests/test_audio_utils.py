"""Slice A1: audio utilities (adapt_audio, pad_to_multiple, load_audio)."""
import math

import numpy as np
import soundfile as sf
import torch


def test_resample_sine():
    from audio_codec_wrappers.audio import adapt_audio
    sr_in, sr_out, f0, amp = 44100, 48000, 1000.0, 0.5
    n = torch.arange(sr_in, dtype=torch.float64)
    x = (amp * torch.sin(2 * math.pi * f0 * n / sr_in)).float()  # 1 s
    y = adapt_audio(x, sr_in, sr_out, 1)
    assert y.shape == (1, 1, sr_out)
    spec = torch.fft.rfft(y[0, 0].double()).abs()
    peak_hz = int(spec.argmax()) * sr_out / y.shape[-1]  # 1 Hz bins
    assert abs(peak_hz - f0) <= 1.0
    rms_ratio = float(y.double().pow(2).mean().sqrt() / x.double().pow(2).mean().sqrt())
    assert abs(rms_ratio - 1.0) <= 0.01


def test_channel_map():
    from audio_codec_wrappers.audio import adapt_audio
    g = torch.Generator().manual_seed(0)
    mono = torch.randn(1, 1, 800, generator=g)
    st = adapt_audio(mono, 8000, 8000, 2)
    assert st.shape == (1, 2, 800)
    assert torch.equal(st[:, 0], mono[:, 0]) and torch.equal(st[:, 1], mono[:, 0])

    stereo = torch.randn(1, 2, 800, generator=g)
    mo = adapt_audio(stereo, 8000, 8000, 1)
    assert mo.shape == (1, 1, 800)
    assert torch.allclose(mo[:, 0], stereo.mean(dim=1), atol=0, rtol=0)

    for shape, want in [((800,), (1, 1, 800)), ((1, 800), (1, 1, 800)), ((3, 1, 800), (3, 1, 800))]:
        assert adapt_audio(torch.zeros(shape), 8000, 8000, 1).shape == want
    assert adapt_audio(torch.zeros(2, 800), 8000, 8000, 2).shape == (1, 2, 800)


def test_pad_to_multiple():
    from audio_codec_wrappers.audio import pad_to_multiple
    for n, want in [(0, 0), (1, 8), (7, 8), (8, 8), (9, 16)]:
        x = torch.ones(1, 2, n)
        y = pad_to_multiple(x, 8)
        assert y.shape == (1, 2, want)
        assert torch.equal(y[..., :n], x)
        assert torch.count_nonzero(y[..., n:]) == 0


def test_load_audio_slice(tmp_path):
    from audio_codec_wrappers.audio import load_audio
    sr = 48000
    rng = np.random.default_rng(0)
    data = rng.uniform(-0.5, 0.5, size=(3 * sr, 2)).astype(np.float32)
    path = tmp_path / "x.wav"
    sf.write(str(path), data, sr, subtype="FLOAT")
    x, got_sr = load_audio(path, 1.0, 2.5)
    assert got_sr == sr
    assert x.shape == (2, 72000)
    assert torch.equal(x, torch.from_numpy(data[48000:120000].T.copy()))
