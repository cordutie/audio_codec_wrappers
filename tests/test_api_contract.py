"""Slices A1-A5: the common API contract, parametrised over every registered codec (design S1 (i)-(vi), D3, D19).

At A1 only the toy_* parametrisations are expected to pass; the real codecs are checked in A2-A5
(`-k spectrostream`, `-k encodec24`, `-k encodec48`, `-k codicodec`).
"""
import functools

import pytest
import torch

from conftest import (ALL_CODECS, CAUSAL_CODECS, CODICODEC_SEEDS, NONCAUSAL_CODECS, chunk_sizes, load_guitar,
                      max_abs, stream)


@functools.lru_cache(maxsize=None)
def _roundtrip(name: str, seed: int = 0):
    """Guitar at the codec rate x [1, C, N'], its encoding, and decode(z, scale, seed) [1, C, T*h]."""
    from audio_codec_wrappers import load_codec
    codec = load_codec(name, device="cuda")
    x, sr = load_guitar()
    xp = codec.prepare(x, sr)
    enc = codec.encode(x, sr)
    y = codec.decode(enc.z, enc.scale, seed=seed)
    return xp, enc, y


def _seeds(name):
    return CODICODEC_SEEDS if name == "codicodec" else [0]


@pytest.mark.parametrize("name", ALL_CODECS)
def test_encode_shapes(name, get_codec):
    codec = get_codec(name)
    info = codec.info
    xp, enc, _ = _roundtrip(name)
    n = xp.shape[-1]
    assert xp.shape[:2] == (1, info.channels)
    assert enc.num_samples == n
    assert enc.z.shape == (1, info.latent_dim, info.num_frames(n))
    assert enc.z.dtype == torch.float32
    assert enc.z.requires_grad is False
    if info.normalizes_input:
        assert enc.scale is not None and enc.scale.shape == (1,) and enc.scale.dtype == torch.float32
    else:
        assert enc.scale is None


@pytest.mark.parametrize("name", ALL_CODECS)
def test_decode_length(name, get_codec):
    info = get_codec(name).info
    _, enc, y = _roundtrip(name)
    assert y.shape == (1, info.channels, enc.z.shape[-1] * info.hop_length)
    assert y.dtype == torch.float32


@pytest.mark.parametrize("name", ALL_CODECS)
def test_alignment(name, get_codec):
    from audio_codec_wrappers.metrics import envelope_lag
    info = get_codec(name).info
    xp, _, y = _roundtrip(name)
    n = xp.shape[-1]
    lag = envelope_lag(xp[0].cpu(), y[0, :, :n].cpu(), info.sample_rate)
    assert abs(lag) <= 256, f"{name}: envelope lag {lag} samples"


@pytest.mark.parametrize("name", ALL_CODECS)
def test_fidelity_beats_noise(name, get_codec):
    """F1 (locked): the reconstruction is closer to the input than RMS-matched noise, rho_nu < 1."""
    from audio_codec_wrappers.metrics import fidelity_report
    info = get_codec(name).info
    for seed in _seeds(name):
        xp, _, y = _roundtrip(name, seed)
        n = xp.shape[-1]
        rep = fidelity_report(xp[0].cpu(), y[0, :, :n].cpu(), info.sample_rate, info, noise_seed=0)
        assert rep["rho_nu"] < 1.0, f"{name} seed {seed}: rho_nu = {rep['rho_nu']}"


@pytest.mark.parametrize("name", ALL_CODECS)
def test_envelope_timing(name, get_codec):
    """F3: on the non-stationary guitar, c(x, x_hat) >= c(x, x delayed by delta) - 0.005 (tie tolerance, Esteban 2026-10-21)."""
    from audio_codec_wrappers.metrics import fidelity_report
    info = get_codec(name).info
    want_delta = 0.25 if name == "codicodec" else 0.05
    for seed in _seeds(name):
        xp, _, y = _roundtrip(name, seed)
        n = xp.shape[-1]
        rep = fidelity_report(xp[0].cpu(), y[0, :, :n].cpu(), info.sample_rate, info, noise_seed=0)
        assert rep["nonstationary"] is True
        assert abs(rep["delta_s"] - want_delta) <= 1e-12
        # Esteban 2026-10-21: tie tolerance (EnCodec 48k missed the strict bar by 0.001)
        assert rep["c_hat"] >= rep["c_delayed"] - 0.005, f"{name} seed {seed}: {rep['c_hat']} < {rep['c_delayed']} - 0.005"


@pytest.mark.parametrize("name", ALL_CODECS)
def test_aliases(name, get_codec):
    codec = get_codec(name)
    x, sr = load_guitar(0.0, 2.0)
    enc = codec.encode(x, sr)
    for alias in (codec.encode_continuous, codec.audio_to_latents):
        other = alias(x, sr)
        assert torch.equal(other.z, enc.z)
        assert (other.scale is None) == (enc.scale is None)
        if enc.scale is not None:
            assert torch.equal(other.scale, enc.scale)
        assert other.num_samples == enc.num_samples
    a = codec.decode(enc.z, enc.scale, seed=0)
    b = codec.latents_to_audio(enc.z, enc.scale, seed=0)
    assert max_abs(a, b) <= 1e-6


@pytest.mark.parametrize("name", ALL_CODECS)
def test_batch_encode(name, get_codec):
    codec = get_codec(name)
    a, sr = load_guitar(0.0, 2.0)
    b, _ = load_guitar(4.0, 6.0)
    batch = torch.cat([a, b], dim=0)  # [2, 1, N]
    eb = codec.encode(batch, sr)
    for i, item in enumerate((a, b)):
        ei = codec.encode(item, sr)
        assert max_abs(eb.z[i:i + 1], ei.z) <= 1e-4
        if ei.scale is not None:
            assert max_abs(eb.scale[i:i + 1], ei.scale) <= 1e-4


@pytest.mark.parametrize("name", CAUSAL_CODECS)
def test_streaming_chunk_invariance(name, get_codec):
    codec = get_codec(name)
    info = codec.info
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    dec = codec.streaming_decoder(batch_size=1)
    L = dec.latency_samples
    ref = stream(dec, z, [T])  # one-shot push
    assert ref.shape == (1, info.channels, T * info.hop_length + L)
    for kind in ("1", "7", "random"):
        out = stream(dec, z, chunk_sizes(kind, T))
        assert out.shape == ref.shape
        assert max_abs(out, ref) <= 1e-4, f"{name}: chunking {kind}"


@pytest.mark.parametrize("name", NONCAUSAL_CODECS)
def test_streaming_unavailable(name, get_codec):
    with pytest.raises(NotImplementedError):
        get_codec(name).streaming_decoder()


@pytest.mark.parametrize("name", CAUSAL_CODECS)
def test_window_decoder_full_window(name, get_codec):
    """Design D19: a window that covers the whole past equals the streaming decoder (same latency, same samples)."""
    codec = get_codec(name)
    info = codec.info
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    sdec = codec.streaming_decoder(batch_size=1)
    wdec = codec.window_decoder(window_frames=T, batch_size=1)
    L = sdec.latency_samples
    assert wdec.latency_samples == L
    sizes = chunk_sizes("7", T)
    ref = stream(sdec, z, sizes)
    out = stream(wdec, z, sizes)
    assert out.shape == ref.shape == (1, info.channels, T * info.hop_length + L)
    assert max_abs(out[..., L:], ref[..., L:]) <= 1e-4, name


@pytest.mark.parametrize("name", NONCAUSAL_CODECS)
def test_window_decoder_unavailable(name, get_codec):
    with pytest.raises(NotImplementedError):
        get_codec(name).window_decoder(window_frames=8)


def test_window_decoder_toy_zero_window(get_codec):
    """toy_causal is frame-wise (receptive field 0): even a window of 0 past frames equals the stream exactly."""
    codec = get_codec("toy_causal")
    x, sr = load_guitar(0.0, 1.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    sizes = chunk_sizes("7", T)
    ref = stream(codec.streaming_decoder(batch_size=1), z, sizes)
    out = stream(codec.window_decoder(window_frames=0, batch_size=1), z, sizes)
    assert max_abs(out, ref) <= 1e-6


def test_window_decoder_encodec24k_short_window_differs(get_codec):
    """Non-vacuity for D19: EnCodec 24k has memory (convolution tails, LSTM), so a window of 0 past frames is NOT the stream."""
    codec = get_codec("encodec24k")
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    sizes = chunk_sizes("7", T)
    ref = stream(codec.streaming_decoder(batch_size=1), z, sizes)
    out = stream(codec.window_decoder(window_frames=0, batch_size=1), z, sizes)
    assert out.shape == ref.shape
    assert max_abs(out, ref) > 1e-3


def test_window_decoder_spectrostream_finite_memory(get_codec):
    """R2-C6: SpectroStream's decoder memory is finite. Measure it empirically (perturb latent frame j, find the last streamed
    sample that changes, R = frames of influence after j), then a window of R + lookahead + 1 frames reproduces the stream."""
    import math
    codec = get_codec("spectrostream")
    info = codec.info
    h, look = info.hop_length, info.decoder_lookahead_frames
    x, sr = load_guitar(0.0, 4.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    dec = codec.streaming_decoder(batch_size=1)
    L = dec.latency_samples
    j = T // 4
    g = torch.Generator().manual_seed(0)
    z2 = z.clone()
    z2[..., j] += torch.randn(z2[..., j].shape, generator=g).to(z2.device)
    a = stream(dec, z, [T])[..., L:L + T * h]
    b = stream(dec, z2, [T])[..., L:L + T * h]
    changed = torch.nonzero(((a - b).abs() > 1e-6).any(dim=1)[0]).flatten()
    assert changed.numel() > 0  # non-vacuous
    first, last = int(changed.min()), int(changed.max())
    assert first >= (j - look) * h  # causal: nothing changes before frame j - lookahead
    assert last < (T - 1) * h, "the influence of one frame reaches the end of the clip: memory not finite within 4 s"
    R = math.ceil((last + 1) / h) - j
    print(f"SpectroStream decoder memory: R = {R} frames ({R * h / info.sample_rate:.3f} s)")
    w = codec.window_decoder(window_frames=R + look + 1, batch_size=1)
    sizes = chunk_sizes("7", T)
    ref = stream(dec, z, sizes)
    out = stream(w, z, sizes)
    assert max_abs(out[..., L:], ref[..., L:]) <= 1e-4


@pytest.mark.parametrize("name", CAUSAL_CODECS)
def test_causality(name, get_codec):
    """Perturbing frames >= k + lookahead leaves the first k*h output samples unchanged."""
    codec = get_codec(name)
    info = codec.info
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z
    T = z.shape[-1]
    k = T // 2
    g = torch.Generator().manual_seed(0)
    z2 = z.clone()
    z2[..., k + info.decoder_lookahead_frames:] += torch.randn(z2[..., k + info.decoder_lookahead_frames:].shape,
                                                               generator=g).to(z2.device)
    dec = codec.streaming_decoder(batch_size=1)
    L = dec.latency_samples
    o1 = stream(dec, z, [T])[..., L:L + k * info.hop_length]
    o2 = stream(dec, z2, [T])[..., L:L + k * info.hop_length]
    assert max_abs(o1, o2) <= 1e-6
    # The perturbation must reach the output after the cut, or the test is vacuous.
    p1 = stream(dec, z, [T])[..., L:]
    p2 = stream(dec, z2, [T])[..., L:]
    assert max_abs(p1, p2) > 1e-3
