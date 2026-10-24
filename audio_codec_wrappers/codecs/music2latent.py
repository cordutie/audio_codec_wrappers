"""music2latent (Sony CSL; consistency-model autoencoder, continuous latents) through the official `music2latent` package.

Rate 44.1 kHz stereo; one latent frame = 4096 samples x 64 channels per audio channel. The wrapper latent stacks the
two channels: rows 0..63 are the left channel, rows 64..127 the right one, so d = 128 and the frame quantum is 1.
Decoding is stochastic (the start noise of the consistency model): `seed` is applied in a forked RNG.
Weights: HF SonyCSLParis/music2latent at the revision pinned below; the sha256 is checked against weights.lock.json
before the pickle checkpoint is loaded, and the inner model is checked key by key (no missing or unexpected keys).
"""
import contextlib
import dataclasses
import hashlib
import json
import pathlib

import torch

from ..base import CodecWrapper
from ..registry import codec_info

REPO = pathlib.Path(__file__).resolve().parents[2]
REPO_ID = "SonyCSLParis/music2latent"
REVISION = "faab47a447a2b663c77df2969e984e411e1d0c8d"
FILE = "music2latent.pt"
HOP = 4096        # samples per latent frame
PAD = 1536        # zeros appended before the library encoder (3 hops of its own 512-sample STFT hop)
MAX_ROWS = 512    # rows (items x channels) per encoder call

# the library would download its own copy into the package directory in __init__; we give it ours
import music2latent.inference as m2l_inference  # noqa: E402
m2l_inference.download_model = lambda: None
# The library runs the encoder under fp16 autocast (hparams.mixed_precision). fp16 rounding made the latents of one
# clip differ by up to 4e-3 between a batch and a single call, so the wrapper runs plain fp32 as the other codecs do.
# The flag is read at call time from the namespaces of inference.py and utils.py (decoder).
import music2latent.utils as m2l_utils  # noqa: E402
m2l_inference.mixed_precision = False
m2l_utils.mixed_precision = False


@contextlib.contextmanager
def _full_precision():
    """TF32 off. With TF32 on (cuDNN convs default), the latent of one clip differed by 7e-4 between a batch call and a
    single call; with it off the difference is 4e-6 (slice 2 check)."""
    conv, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = conv, mm


def verify_state_dict(module: torch.nn.Module, state_dict: dict) -> None:
    """load_state_dict(strict=True) semantics without touching the module: RuntimeError naming the keys."""
    have = set(module.state_dict().keys())
    want = set(state_dict.keys())
    missing, unexpected = sorted(have - want), sorted(want - have)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch; missing keys: {missing}; unexpected keys: {unexpected}")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _checkpoint_path() -> tuple[pathlib.Path, str]:
    """Path of music2latent.pt (pinned revision) and the sha256 recorded in weights.lock.json."""
    from huggingface_hub import hf_hub_download
    lock = REPO / "weights.lock.json"
    entry = json.loads(lock.read_text()).get("music2latent", {}).get("codec", {}).get("files", {}).get(FILE)
    if entry is None:
        raise FileNotFoundError("music2latent is not in weights.lock.json")
    return pathlib.Path(hf_hub_download(REPO_ID, FILE, revision=REVISION)), entry["sha256"]


class Music2LatentWrapper(CodecWrapper):
    def __init__(self, device="cuda"):
        path, want = _checkpoint_path()
        # the checkpoint is a pickle: load it only if it is the file recorded in the lock
        have = _sha256(path)
        if have != want:
            raise RuntimeError(f"{FILE} sha256 {have} differs from weights.lock.json {want}")
        self.info = dataclasses.replace(codec_info("music2latent"), weights=f"hf:{REPO_ID}@{REVISION[:8]}")
        self.device = torch.device(device)
        self.encdec = m2l_inference.EncoderDecoder(load_path_inference=str(path), device=self.device)
        # the library loads with strict=False; check the same file strictly
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        verify_state_dict(self.encdec.gen, ckpt["gen_state_dict"])
        del ckpt
        self.encdec.gen.eval()
        for p in self.encdec.gen.parameters():
            p.requires_grad_(False)

    def _encode(self, audio):
        # audio [B, 2, M], M a multiple of 4096. The library treats rows as independent channels, so the items and
        # channels become rows [2B, M + 1536] and one call encodes all of them (build throughput).
        B, C, M = audio.shape
        rows = torch.cat([audio.reshape(B * C, M), torch.zeros(B * C, PAD, device=audio.device)], dim=-1)
        with _full_precision():
            z = self.encdec.encode(rows, max_batch_size=min(B * C, MAX_ROWS))    # [2B, 64, M/4096]
        z = torch.as_tensor(z).float().to(self.device)
        return z.reshape(B, C * 64, M // HOP), None

    def decode(self, z, scale=None, seed=0):
        z = z.to(self.device).float()
        T = z.shape[-1]
        outs = []
        # per item, the seed is applied in a forked RNG, so item b decodes the same alone or in a batch and the
        # caller's global state is untouched
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.no_grad(), _full_precision():
            for zi in z:
                with torch.random.fork_rng(devices=devices, enabled=seed is not None):
                    if seed is not None:
                        torch.manual_seed(int(seed))
                        if self.device.type == "cuda":
                            torch.cuda.manual_seed_all(int(seed))
                    y = self.encdec.decode(zi.reshape(2, 64, T), denoising_steps=1)    # [2, samples]
                outs.append(torch.as_tensor(y).float().to(self.device)[None, :, : T * HOP])
        return torch.cat(outs, dim=0)
