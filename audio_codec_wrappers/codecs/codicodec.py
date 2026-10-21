"""CoDiCodec (Sony CSL; consistency-model decoder, continuous latents) through the official `codicodec` package (plan A5; C-3, R1-C4; D8).

Rate 44.1 kHz stereo; one chunk = 32 768 samples = 8 latent frames x 64 channels, so the frame quantum is 8.
The 8 frames of a chunk are a fixed reshape of 128 summary embeddings of dimension 4; they are not temporally ordered.
Latent = library encode (atanh(.)/0.8 applied by the library). Decoding is stochastic: `seed` is applied in a forked RNG.
Weights: scripts/fetch_weights.py --codec codicodec (pickle checkpoint, official HF repo, sha256 checked before loading).
"""
import dataclasses
import hashlib
import json
import os
import pathlib

import torch

from ..audio import adapt_audio
from ..base import CodecWrapper
from ..registry import codec_info

REPO = pathlib.Path(__file__).resolve().parents[2]
CHUNK = 32768  # samples per chunk at 44.1 kHz


def _flags() -> tuple:
    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)


def _set_flags(flags: tuple) -> None:
    (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) = flags


# The codicodec package sets these global flags when it is imported; they are restored after the import
_before = _flags()
import codicodec.inference as cc_inference  # noqa: E402
_set_flags(_before)
# The library runs the encoder and decoder under fp16 autocast (hparams.mixed_precision). fp16 rounding made the
# latents of equal audio differ by up to 2e-2 after atanh/0.8, so the wrapper runs plain fp32 (with the process's
# own TF32 flags, i.e. as the other codecs do). utils.mixed_precision is read at call time by distribute().
import codicodec.utils as cc_utils  # noqa: E402
cc_utils.mixed_precision = False


def chunks_to_latent(x: torch.Tensor) -> torch.Tensor:
    """Library layout [n, 8, 64] -> wrapper latent [1, 64, 8n] (frame 8i+j = chunk i, row j)."""
    n = x.shape[0]
    return x.reshape(1, n * 8, 64).transpose(1, 2).contiguous()


def latent_to_chunks(z: torch.Tensor) -> torch.Tensor:
    """Wrapper latent [1, 64, 8n] -> library layout [n, 8, 64]."""
    return z.transpose(1, 2).reshape(-1, 8, 64).contiguous()


def _checkpoint_path() -> tuple[pathlib.Path, str]:
    """Cache path of codicodec.pt and the sha256 recorded in weights.lock.json."""
    root = pathlib.Path(os.environ.get("ACW_CACHE", pathlib.Path.home() / ".cache" / "audio_codec_wrappers"))
    path = root / "codicodec" / "codicodec.pt"
    lock = REPO / "weights.lock.json"
    msg = "codicodec weights not found; run: python scripts/fetch_weights.py --codec codicodec"
    if not path.exists() or not lock.exists():
        raise FileNotFoundError(msg)
    entry = json.loads(lock.read_text()).get("codicodec", {}).get("codec", {}).get("files", {}).get("codicodec.pt")
    if entry is None:
        raise FileNotFoundError(msg)
    return path, entry["sha256"]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class CoDiCodecCodec(CodecWrapper):
    def __init__(self, device="cuda"):
        path, want = _checkpoint_path()
        # the checkpoint is a pickle: load it only if it is the file recorded at fetch time
        have = _sha256(path)
        if have != want:
            raise ValueError(f"codicodec.pt sha256 {have} differs from weights.lock.json {want}; fetch it again")
        self.info = dataclasses.replace(codec_info("codicodec"), weights=f"hf:SonyCSLParis/codicodec@{want[:12]}")
        self.device = torch.device(device)
        before = _flags()
        try:
            # the library would download its own copy into the package directory in __init__; we give it ours
            cc_inference.download_model = lambda: None
            self.encdec = cc_inference.EncoderDecoder(load_path_inference=str(path), device=self.device)
        finally:
            _set_flags(before)
        for p in self.encdec.gen.parameters():
            p.requires_grad_(False)

    def prepare(self, audio, sample_rate):
        # Resample on the CPU. The latent is very sensitive to the input (a 1e-7 perturbation of the audio moved a
        # latent by 1.3e-2 on the guitar; hypothesis: atanh amplifies values near the FSQ bound), so the resampled
        # input must not depend on whether the caller or the wrapper resamples, nor on the device.
        x = adapt_audio(torch.as_tensor(audio).cpu(), sample_rate, self.info.sample_rate, self.info.channels)
        return x.to(self.device)

    def _encode(self, audio):
        # audio [B, 2, M], M a multiple of 32768; one item at a time ([2, M] -> [n, 8, 64])
        zs = []
        for a in audio:
            ref = self.encdec.encode(a.cpu().numpy())
            zs.append(chunks_to_latent(torch.as_tensor(ref).float().to(self.device)))
        return torch.cat(zs, dim=0), None

    def decode(self, z, scale=None, seed=0):
        z = z.to(self.device).float()
        outs = []
        # the seed is applied in a forked RNG so the caller's global state is untouched
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
            for zi in z:
                y = self.encdec.decode(latent_to_chunks(zi[None]))
                # the library returns 1024 samples more than the 32 768 per chunk (STFT tail); keep T*h
                outs.append(torch.as_tensor(y).float().to(self.device)[None, :, : zi.shape[-1] * self.info.hop_length])
        return torch.cat(outs, dim=0)
