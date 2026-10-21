"""Slice A4: EnCodec 48k wrapper equals the legacy processor (existing behaviour, P-9, D14)."""
import functools
import importlib.util

import numpy as np
import torch
import torchaudio

from conftest import REPO, load_guitar, max_abs


@functools.lru_cache(maxsize=None)
def _legacy():
    """processors/wrapper_encodec.py imported by file path (the legacy folder is not a package)."""
    spec = importlib.util.spec_from_file_location("legacy_wrapper_encodec", REPO / "processors" / "wrapper_encodec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EncodecProcessor(sr=48000, streamable=True, device="cuda")


def _segment() -> torch.Tensor:
    """0.32 s (15,360 samples) of guitar resampled to 48 kHz, duplicated to stereo: [2, 15360]."""
    x, sr = load_guitar(0.0, 1.0)
    x48 = torchaudio.functional.resample(x[0], sr, 48000)
    return x48.repeat(2, 1)[:, :15360].contiguous()


def test_encode_matches_legacy(get_codec):
    seg = _segment()
    lat_list, meta = _legacy().audio_to_latents(seg, 48000)
    z_ref = lat_list[0]
    s_ref = meta["audio_scales"][0].reshape(-1)
    enc = get_codec("encodec48k").encode(seg.unsqueeze(0), 48000)
    assert enc.z.shape == (1, 128, 48) and z_ref.shape == (1, 128, 48)
    assert max_abs(enc.z, z_ref) <= 1e-6
    assert max_abs(enc.scale, s_ref) <= 1e-6


def test_decode_matches_legacy(get_codec):
    seg = _segment()
    codec = get_codec("encodec48k")
    enc = codec.encode(seg.unsqueeze(0), 48000)
    y = codec.decode(enc.z, enc.scale)
    y_ref = _legacy().decode_latents_audio(enc.z, {"audio_scales": [enc.scale.reshape(-1, 1)]})
    assert y.shape == (1, 2, 15360)
    assert max_abs(y, y_ref) <= 1e-6


def test_scale_known(get_codec):
    seg = _segment()
    enc = get_codec("encodec48k").encode(seg.unsqueeze(0), 48000)
    mono = seg.numpy().astype(np.float64).mean(axis=0)
    want = float(np.sqrt(np.mean(mono ** 2)) + 1e-8)
    assert abs(float(enc.scale[0]) - want) <= 1e-6 * want
