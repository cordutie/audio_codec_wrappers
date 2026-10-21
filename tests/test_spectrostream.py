"""Slice A2: SpectroStream through magenta-rt's JAX code (RQ2; C-1; design A1, A2, D16).

Interface used beyond the common API (plan, binding notes for `spectrostream`):
- codec.module: the flax SpectroStream module built from stft_spectrostream_40ms_generic_48khz_stereo_config();
- codec.variables: the variables dict passed to codec.module.apply.
"""
import gc

import torch
import torchaudio

from conftest import chunk_sizes, load_guitar, max_abs, stream


def _guitar48_stereo(seconds: float) -> torch.Tensor:
    """[1, 2, seconds*48000] at 48 kHz, from guitar (44.1 kHz mono)."""
    x, sr = load_guitar()
    x48 = torchaudio.functional.resample(x[0], sr, 48000).repeat(2, 1)
    n = int(round(seconds * 48000))
    assert x48.shape[-1] >= n
    return x48[:, :n].unsqueeze(0).contiguous()


def test_shapes_rate(get_codec):
    codec = get_codec("spectrostream")
    x = _guitar48_stereo(10.0)
    assert x.shape[-1] == 10 * 1920 * 25
    enc = codec.encode(x, 48000)
    assert enc.z.shape == (1, 256, 250)


def test_latent_is_waveform_to_embeddings(get_codec):
    import jax
    import jax.numpy as jnp
    import sequence_layers.jax as sl
    codec = get_codec("spectrostream")
    x = _guitar48_stereo(4.0)
    z = codec.encode(x, 48000).z  # [1, 256, 100]
    module = codec.module
    method = type(module).waveform_to_embeddings

    @jax.jit
    def ref_fn(variables, values):
        seq = sl.Sequence.from_values(values)
        return module.apply(variables, seq, training=False, method=method).values

    values = jnp.asarray(x[0].T.numpy())[None]  # [1, N, 2]
    ref = torch.from_numpy(jax.device_get(ref_fn(codec.variables, values)).copy())  # [1, T, 256]
    ref = ref.transpose(1, 2)
    scale = max(1.0, float(ref.abs().max()))
    assert max_abs(z, ref) <= 1e-6 * scale


def test_stream_equals_full(get_codec):
    codec = get_codec("spectrostream")
    x = _guitar48_stereo(4.0)
    z = codec.encode(x, 48000).z
    T = z.shape[-1]
    full = codec.decode(z)
    assert full.shape == (1, 2, T * 1920)
    dec = codec.streaming_decoder(batch_size=1)
    L = dec.latency_samples
    for kind in ("1", "8"):
        out = stream(dec, z, chunk_sizes(kind, T))
        assert out.shape == (1, 2, T * 1920 + L)
        assert max_abs(out[..., L:L + T * 1920], full) <= 1e-4, f"push size {kind}"


def test_latency_reported(get_codec):
    import magenta_rt.jax.spectrostream as ss
    import sequence_layers.jax as sl
    from sequence_layers.jax import utils
    cfg = ss.stft_spectrostream_40ms_generic_48khz_stereo_config()
    lib_latency = int(utils.get_output_latency(sl.Serial.Config((cfg.decoder, cfg.inverse_stft))))
    L = get_codec("spectrostream").streaming_decoder().latency_samples
    print("spectrostream decoder latency L =", L, "samples (library:", lib_latency, ")")
    assert L == lib_latency
    assert L >= 1920


def test_jax_does_not_starve_torch(get_codec):
    codec = get_codec("spectrostream")
    x = _guitar48_stereo(10.0)
    enc = codec.encode(x, 48000)
    y = codec.decode(enc.z)
    assert torch.isfinite(y).all()
    del enc, y
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    assert free >= 0.5 * total, f"free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB"
    big = torch.empty(2 ** 31, device="cuda")  # 8 GiB of float32
    assert big.numel() == 2 ** 31
    del big
    torch.cuda.empty_cache()
