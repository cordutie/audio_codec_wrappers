"""Slice A5: CoDiCodec wrapper (C-3, R1-C4; design "CoDiCodec specifics", D8).

Interface used beyond the common API (plan, binding notes for `codicodec`):
- module functions chunks_to_latent([n, 8, 64]) -> [1, 64, 8n] and latent_to_chunks([1, 64, 8n]) -> [n, 8, 64];
- codec.encdec: the codicodec.EncoderDecoder instance the wrapper uses.
"""
import subprocess
import sys

import torch
import torchaudio

from conftest import REPO, load_guitar, max_abs

C = 32768  # samples per chunk at 44.1 kHz


def test_layout_known_answer():
    from audio_codec_wrappers.codecs.codicodec import chunks_to_latent, latent_to_chunks
    n = 3
    x = torch.arange(n * 8 * 64, dtype=torch.float32).reshape(n, 8, 64)
    z = chunks_to_latent(x)
    assert z.shape == (1, 64, 8 * n)
    for i in range(n):
        for j in range(8):
            assert torch.equal(z[0, :, 8 * i + j], x[i, j, :])
    assert torch.equal(latent_to_chunks(z), x)


def test_matches_library_encode(get_codec):
    from audio_codec_wrappers.audio import pad_to_multiple
    from audio_codec_wrappers.codecs.codicodec import chunks_to_latent
    codec = get_codec("codicodec")
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z
    xs = pad_to_multiple(x[0].repeat(2, 1), C)  # stereo [2, N] padded like the wrapper
    ref = codec.encdec.encode(xs.numpy())  # [n, 8, 64], atanh(.)/0.8 applied by the library
    ref = torch.as_tensor(ref).float().cpu()
    assert ref.shape[1:] == (8, 64)
    assert max_abs(z, chunks_to_latent(ref)) <= 1e-6


def test_seed_control(get_codec):
    codec = get_codec("codicodec")
    x, sr = load_guitar(0.0, 3.0)
    z = codec.encode(x, sr).z  # 5 chunks
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    a = codec.decode(z, seed=0)
    b = codec.decode(z, seed=0)
    c = codec.decode(z, seed=1)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    assert max_abs(a, b) <= 1e-6
    assert max_abs(a, c) >= 1e-3


def test_resamples_to_44k(get_codec):
    codec = get_codec("codicodec")
    x, sr = load_guitar(0.0, 3.0)
    x48 = torchaudio.functional.resample(x, sr, 48000)
    a = codec.encode(x48, 48000).z
    b = codec.encode(torchaudio.functional.resample(x48, 48000, 44100), 44100).z
    assert a.shape == b.shape
    assert max_abs(a, b) <= 1e-5


def test_chunk_locality(get_codec):
    """Chunk i depends on samples [c*i, c*(i+1) + 1024) (STFT window 2048, hop 1024, C-3, R1-C4)."""
    codec = get_codec("codicodec")
    x, sr = load_guitar()
    assert sr == 44100
    z = codec.encode(x[..., : 4 * C], sr).z
    assert z.shape[-1] == 32
    for i in (1, 2):
        zi = codec.encode(x[..., C * i: C * (i + 1) + 1024], sr).z
        assert max_abs(zi[..., :8], z[..., 8 * i: 8 * i + 8]) <= 1e-4


def test_frame_count(get_codec):
    codec = get_codec("codicodec")
    g = torch.Generator().manual_seed(0)
    for n, want in [(32768, 8), (32769, 16), (65536, 16)]:
        x = 0.1 * torch.randn(1, 2, n, generator=g)
        assert codec.encode(x, 44100).z.shape == (1, 64, want)


def test_global_torch_flags_unchanged():
    """The codicodec package sets cudnn.benchmark and TF32 flags at import (inference.py:14-19);
    the wrapper must restore them so other codecs' numerics do not depend on test or import order."""
    code = (
        "import torch\n"
        "def flags():\n"
        "    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,\n"
        "            torch.backends.cuda.matmul.allow_tf32,\n"
        "            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)\n"
        "before = flags()\n"
        "from audio_codec_wrappers import load_codec\n"
        "load_codec('codicodec', device='cuda')\n"
        "after = flags()\n"
        "assert before == after, (before, after)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "ok"
