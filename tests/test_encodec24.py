"""Slice A3: EnCodec 24k wrapper and its hand-written stateful decoder (C-4, design R1-C3, S1 (v))."""
import copy
import functools

import torch

from conftest import chunk_sizes, load_guitar, max_abs, stream


@functools.lru_cache(maxsize=None)
def _official():
    """The official package's model, loaded independently of the wrapper (same pretrained weights)."""
    from encodec import EncodecModel
    return EncodecModel.encodec_model_24khz().cuda().eval()


def _guitar24(codec, start=0.0, end=None):
    from audio_codec_wrappers.audio import pad_to_multiple
    x, sr = load_guitar(start, end)
    return x, sr, pad_to_multiple(codec.prepare(x, sr), 320)


def test_latent_is_encoder_output(get_codec):
    codec = get_codec("encodec24k")
    x, sr, x24 = _guitar24(codec)
    z = codec.encode(x, sr).z
    with torch.no_grad():
        ref = _official().encoder(x24.cuda())
    assert max_abs(z, ref) <= 1e-6


def test_decode_official_is_official(get_codec):
    codec = get_codec("encodec24k")
    x, sr, _ = _guitar24(codec)
    z = codec.encode(x, sr).z
    with torch.no_grad():
        ref = _official().decoder(z)
    assert max_abs(codec.decode_official(z), ref) <= 1e-6


def test_decode_equals_stream(get_codec):
    codec = get_codec("encodec24k")
    x, sr, _ = _guitar24(codec, 0.0, 3.0)
    z = codec.encode(x, sr).z
    full = codec.decode(z)
    dec = codec.streaming_decoder(batch_size=1)
    assert dec.latency_samples == 0
    for kind in ("1", "7", "random"):
        out = stream(dec, z, chunk_sizes(kind, z.shape[-1]))
        assert out.shape == full.shape
        assert max_abs(out, full) <= 1e-4, f"chunking {kind}"


def test_stream_exact_cpu():
    from audio_codec_wrappers import load_codec
    from audio_codec_wrappers.codecs.encodec24 import EncodecStreamingDecoder
    codec = load_codec("encodec24k", device="cpu")
    x, sr = load_guitar(0.0, 2.0)
    z = codec.encode(x, sr).z
    assert z.device.type == "cpu"
    full = codec.decode(z)
    dec = codec.streaming_decoder(batch_size=1)
    for kind in ("1", "7", "random"):
        out = stream(dec, z, chunk_sizes(kind, z.shape[-1]))
        assert max_abs(out, full) <= 1e-5, f"fp32 chunking {kind}"
    # float64 mirror: a double copy of the official decoder, one-shot push vs chunked pushes
    dec64 = EncodecStreamingDecoder(copy.deepcopy(codec.model.decoder).double(), batch_size=1)
    z64 = z.double()
    one = stream(dec64, z64, [z64.shape[-1]])
    assert one.dtype == torch.float64
    for kind in ("1", "7", "random"):
        out = stream(dec64, z64, chunk_sizes(kind, z64.shape[-1]))
        assert max_abs(out, one) <= 1e-10, f"fp64 chunking {kind}"


def test_official_difference_reported(get_codec):
    """Reported, not a criterion: |decode - decode_official| per 0.1 s window (start transient of R1-C3)."""
    codec = get_codec("encodec24k")
    x, sr, _ = _guitar24(codec)
    z = codec.encode(x, sr).z
    d = (codec.decode(z) - codec.decode_official(z)).abs().amax(dim=(0, 1)).cpu()
    win = 2400  # 0.1 s at 24 kHz
    per_window = [float(d[i:i + win].max()) for i in range(0, d.shape[-1] - win + 1, win)]
    print("encodec24k max|decode - decode_official| per 0.1 s:", ["%.2e" % v for v in per_window])
    assert all(torch.isfinite(torch.tensor(per_window)))
    assert per_window[-1] <= per_window[0]


def test_no_scale(get_codec):
    codec = get_codec("encodec24k")
    x, sr = load_guitar(0.0, 1.0)
    assert codec.encode(x, sr).scale is None
