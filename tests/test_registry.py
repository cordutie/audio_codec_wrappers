"""Slice A1: registry and static CodecInfo (design D3, plan Interfaces table)."""
import subprocess
import sys

import pytest

from conftest import ALL_CODECS, REPO

# name: (sample_rate, channels, hop, d, q, causal, lookahead, normalizes, stochastic, backend, frame_rate)
TABLE = {
    "encodec24k": (24000, 1, 320, 128, 1, True, 0, False, False, "torch", 75.0),
    "spectrostream": (48000, 2, 1920, 256, 1, True, 1, False, False, "jax", 25.0),
    "encodec48k": (48000, 2, 320, 128, 1, False, None, True, False, "torch", 150.0),
    "toy_causal": (8000, 1, 8, 16, 1, True, 0, False, False, "torch", 1000.0),
    "toy_noncausal": (8000, 1, 8, 16, 1, False, None, True, False, "torch", 1000.0),
}


def test_info_table():
    from audio_codec_wrappers import codec_info, list_codecs
    assert sorted(list_codecs()) == sorted(ALL_CODECS)
    for name, (sr, c, h, d, q, causal, la, norm, stoch, backend, fr) in TABLE.items():
        info = codec_info(name)
        assert info.name == name
        assert (info.sample_rate, info.channels, info.hop_length, info.latent_dim, info.frame_quantum) == (sr, c, h, d, q)
        assert info.causal is causal
        assert info.decoder_lookahead_frames == la
        assert info.normalizes_input is norm
        assert info.stochastic_decoder is stoch
        assert info.backend == backend
        assert info.frame_rate == fr
    # num_frames = q * ceil(N / (q*h)) at the codec rate
    m2l = codec_info("music2latent")
    assert [m2l.num_frames(n) for n in (4096, 4097, 8192)] == [1, 2, 2]
    e24 = codec_info("encodec24k")
    assert [e24.num_frames(n) for n in (1, 320, 321)] == [1, 1, 2]


def test_info_is_light():
    code = (
        "import sys\n"
        "from audio_codec_wrappers import codec_info\n"
        "codec_info('spectrostream'); codec_info('music2latent')\n"
        "heavy = [m for m in ('jax', 'transformers', 'encodec', 'music2latent') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"codec_info imported heavy modules: {out.stdout.strip()}"


def test_unknown():
    from audio_codec_wrappers import codec_info, load_codec
    for fn in (codec_info, load_codec):
        with pytest.raises(KeyError) as e:
            fn("no_such_codec")
        msg = str(e.value)
        for name in ALL_CODECS:
            assert name in msg
