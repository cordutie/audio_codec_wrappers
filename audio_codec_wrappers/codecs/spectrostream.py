"""SpectroStream (Magenta RealTime 2) through magenta-rt's JAX code (RQ2; plan A2; C-1, C-2; design D16).

Latent = encoder output before the RVQ (waveform_to_embeddings); decode bypasses the RVQ (embeddings_to_waveform).
Audio is 48 kHz stereo, latents are 25 Hz x 256. Weights: scripts/fetch_weights.py --codec spectrostream.
"""
import dataclasses
import json
import os
import pathlib

import torch

from .. import configure_jax_env
from ..base import CodecWrapper, StreamingDecoder
from ..registry import codec_info

configure_jax_env()

import magenta_rt  # noqa: E402,F401  (installs the import hook of the vendored sequence_layers)
import flax.core.meta as flax_meta  # noqa: E402
import flax.traverse_util as flaxtu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import safetensors.flax as safetensors_flax  # noqa: E402
import sequence_layers.jax as sl  # noqa: E402
from sequence_layers.jax import utils as sl_utils  # noqa: E402

import magenta_rt.jax.spectrostream as ss  # noqa: E402

from ..jax_bridge import jax_to_torch, torch_to_jax  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
ENCODE_PIECE_SECONDS = 30  # whole-file encode falls back to pieces of this length if the GPU runs out of memory


def _cache_dir() -> pathlib.Path:
    root = pathlib.Path(os.environ.get("ACW_CACHE", pathlib.Path.home() / ".cache" / "audio_codec_wrappers"))
    return root / "spectrostream"


def _weight_error(what: str) -> FileNotFoundError:
    return FileNotFoundError(f"spectrostream weights not found ({what}); run: python scripts/fetch_weights.py --codec spectrostream")


def _nest(flat: dict) -> dict:
    """Flat '/'-joined keys to a nested dict (system.py:171-175)."""
    return flaxtu.unflatten_dict({tuple(k.split("/")): v for k, v in flat.items()})


def load_params() -> dict:
    """Params tree {'params': {'encoder': ..., 'decoder': ...}}.

    Route 1: the codec-only encoder.safetensors and decoder.safetensors (keys start with 'params/').
    Route 2: params['soundstream'] of the full mrt2_base.safetensors checkpoint.
    The route used is the one recorded in weights.lock.json (route 1 if both exist).
    """
    base = _cache_dir()
    enc = base / "resources" / "spectrostream" / "encoder.safetensors"
    dec = base / "resources" / "spectrostream" / "decoder.safetensors"
    full = base / "checkpoints" / "mrt2_base.safetensors"
    if enc.exists() and dec.exists():
        tree = _nest({**safetensors_flax.load_file(str(enc)), **safetensors_flax.load_file(str(dec))})
        # decoder.safetensors also carries the RVQ embedding, which the bypass does not use
        return {"params": {"encoder": tree["params"]["encoder"], "decoder": tree["params"]["decoder"]}}
    if full.exists():
        tree = _nest(safetensors_flax.load_file(str(full)))
        ss_params = tree["params"]["soundstream"]
        return {"params": {"encoder": ss_params["encoder"], "decoder": ss_params["decoder"]}}
    raise _weight_error(str(base))


def _check_tree(module, variables, example):
    """The loaded tree must have the module's structure and leaf shapes; raises with the differences."""
    expected = jax.eval_shape(
        lambda x: module.init(jax.random.PRNGKey(0), sl.Sequence.from_values(x), training=False), example)
    want = {k: v.shape for k, v in flaxtu.flatten_dict(flax_meta.unbox(expected)).items()}
    have = {k: tuple(v.shape) for k, v in flaxtu.flatten_dict(variables).items()}
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    wrong = [k for k in want if k in have and tuple(want[k]) != have[k]]
    if missing or extra or wrong:
        raise ValueError(f"spectrostream params tree mismatch: missing {missing[:5]}, extra {extra[:5]}, shape {wrong[:5]}")


class SpectroStreamStreamingDecoder(StreamingDecoder):
    """Sequence-layers step state of embeddings_to_waveform_layer (decoder + inverse STFT)."""

    def __init__(self, codec: "SpectroStreamCodec", batch_size: int = 1):
        self.codec = codec
        self.batch_size = batch_size
        self.latency_samples = codec.latency_samples
        self.reset()

    def reset(self) -> None:
        self.state = self.codec._init_state(self.batch_size)

    def push(self, z: torch.Tensor) -> torch.Tensor:
        """[B, d, n] -> [B, 2, n*h]"""
        values = torch_to_jax(z.float().transpose(1, 2))  # [B, n, d]
        out, self.state = self.codec._step(self.codec.variables, values, self.state)
        return jax_to_torch(out).transpose(1, 2).contiguous()

    def flush(self) -> torch.Tensor:
        h = self.codec.info.hop_length
        k = -(-self.latency_samples // h)
        zeros = torch.zeros(self.batch_size, self.codec.info.latent_dim, k, device=self.codec.device)
        return self.push(zeros)[..., :self.latency_samples]


class SpectroStreamCodec(CodecWrapper):
    def __init__(self, device):
        self.info = dataclasses.replace(codec_info("spectrostream"), weights=self._weights_id())
        self.device = torch.device(device)
        # the codec without its RVQ: only the encoder and decoder parameters exist
        self.config = dataclasses.replace(ss.stft_spectrostream_40ms_generic_48khz_stereo_config(), quantizer=None)
        self.module = self.config.make()
        self.variables = jax.tree.map(jnp.asarray, load_params())
        example = jax.ShapeDtypeStruct((1, self.info.hop_length, 2), jnp.float32)
        _check_tree(self.module, self.variables, example)
        # the library's output latency of the decoder chain (P-11: 1920 samples)
        self.latency_samples = int(sl_utils.get_output_latency(sl.Serial.Config((self.config.decoder, self.config.inverse_stft))))
        self._build_functions()

    @staticmethod
    def _weights_id() -> str:
        lock = REPO / "weights.lock.json"
        if lock.exists():
            entries = json.loads(lock.read_text()).get("spectrostream", {})
            for route in ("codec", "full"):
                if route in entries:
                    files = entries[route]["files"]
                    digest = "+".join(f["sha256"][:12] for f in files.values())
                    return f"hf:{entries[route]['source'][3:]}@{entries[route]['revision'][:12]}/{route}:{digest}"
        return ""

    def _build_functions(self):
        module = self.module
        to_emb = type(module).waveform_to_embeddings
        to_wav = type(module).embeddings_to_waveform

        # full fp32 convolutions: the default (TF32) differs between one-shot and streaming by ~1e-3
        @jax.jit
        @jax.default_matmul_precision("highest")
        def encode_fn(variables, wav):  # [B, N, 2] -> [B, T, d]
            seq = sl.Sequence.from_values(wav)
            return module.apply(variables, seq, training=False, method=to_emb).values

        @jax.jit
        @jax.default_matmul_precision("highest")
        def decode_fn(variables, emb):  # [B, T, d] -> [B, T*h, 2]
            seq = sl.Sequence.from_values(emb)
            return module.apply(variables, seq, training=False, method=to_wav).values

        d = self.info.latent_dim

        def init_state(variables, batch_size):
            def method(m):
                return m.embeddings_to_waveform_layer.get_initial_state(
                    batch_size, sl.ShapeDType((d,), jnp.float32), training=False)
            return module.apply(variables, method=method)

        @jax.jit
        @jax.default_matmul_precision("highest")
        def step_fn(variables, emb, state):
            def method(m, x, s):
                return m.embeddings_to_waveform_layer.step(sl.Sequence.from_values(x), s, training=False)
            out, state = module.apply(variables, emb, state, method=method)
            return out.values, state

        def init_enc_state(variables, batch_size):
            def method(m):
                return m.waveform_to_embeddings_layer.get_initial_state(
                    batch_size, sl.ShapeDType((2,), jnp.float32), training=False)
            return module.apply(variables, method=method)

        @jax.jit
        @jax.default_matmul_precision("highest")
        def enc_step_fn(variables, wav, state):
            def method(m, x, s):
                return m.waveform_to_embeddings_layer.step(sl.Sequence.from_values(x), s, training=False)
            out, state = module.apply(variables, wav, state, method=method)
            return out.values, state

        self._init_enc_state, self._enc_step = init_enc_state, enc_step_fn
        self._encode_fn, self._decode_fn, self._step = encode_fn, decode_fn, step_fn
        self._init_state_fn = init_state

    def _init_state(self, batch_size: int):
        return self._init_state_fn(self.variables, batch_size)

    def _encode(self, audio):
        wav = torch_to_jax(audio.transpose(1, 2))  # [B, M, 2]
        try:
            emb = self._encode_fn(self.variables, wav)
        except jax.errors.JaxRuntimeError as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            emb = self._encode_in_pieces(wav)
        return jax_to_torch(emb).transpose(1, 2).contiguous(), None

    def _encode_in_pieces(self, wav, piece_seconds: float = ENCODE_PIECE_SECONDS):
        """Streaming-step encode in pieces of whole frames (D16 fallback). wav [B, M, 2], M a multiple of h.

        The encoder's output is delayed by one frame, so one frame of zeros is fed at the end and the first output
        frame is dropped; the result equals the one-shot encode.
        """
        h = self.info.hop_length
        piece = max(1, int(piece_seconds * self.info.sample_rate) // h) * h
        state = self._init_enc_state(self.variables, wav.shape[0])
        padded = jnp.concatenate([wav, jnp.zeros((wav.shape[0], h, 2), wav.dtype)], axis=1)
        outs = []
        for a in range(0, padded.shape[1], piece):
            out, state = self._enc_step(self.variables, padded[:, a:a + piece], state)
            outs.append(out)
        return jnp.concatenate(outs, axis=1)[:, 1:]

    def decode(self, z, scale=None, seed=0):
        # the layer-wise output drops the first L samples (the latency), so zero frames at the end restore the length,
        # exactly what flush() feeds the streaming decoder
        k = -(-self.latency_samples // self.info.hop_length)
        z = torch.cat([z.to(self.device).float(), torch.zeros(z.shape[0], z.shape[1], k, device=self.device)], dim=-1)
        emb = torch_to_jax(z.transpose(1, 2))  # [B, T + k, d]
        wav = self._decode_fn(self.variables, emb)
        return jax_to_torch(wav).transpose(1, 2).contiguous()

    def streaming_decoder(self, batch_size: int = 1) -> StreamingDecoder:
        return SpectroStreamStreamingDecoder(self, batch_size)
