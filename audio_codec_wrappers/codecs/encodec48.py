"""EnCodec 48 kHz (non-causal, stereo, HuggingFace weights) as in the legacy processor (plan A4; design D14, P-9).

Latent = encoder output before the quantizer (128 x 150 Hz). The input is normalised by its RMS (mono mix, +1e-8) and the
decoder output is multiplied back by that scale, as in processors/wrapper_encodec.py with streamable=True (no chunking).
"""
import torch

from ..base import CodecWrapper
from ..registry import codec_info


class Encodec48Codec(CodecWrapper):
    def __init__(self, device="cuda"):
        from transformers import EncodecModel
        self.info = codec_info("encodec48k")
        self.device = torch.device(device)
        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz")
        # whole input at once, no 1 s chunks
        self.model.config.chunk_length_s = None
        self.model.config.overlap = None
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _encode(self, audio):
        # RMS of the mono mix per item, then the encoder on the normalised item (one item at a time, so the result
        # does not depend on the batch)
        mono = audio.sum(dim=1, keepdim=True) / audio.shape[1]
        scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8  # [B, 1, 1]
        z = torch.cat([self.model.encoder(a[None] / s[None]) for a, s in zip(audio, scale)], dim=0)
        return z, scale.reshape(-1)

    def decode(self, z, scale=None, seed=0):
        if scale is None:
            raise ValueError("encodec48k decode needs the scale returned by encode")
        z = z.to(self.device).float()
        scale = scale.to(self.device).float().reshape(-1, 1, 1)
        with torch.no_grad():
            y = torch.cat([self.model.decoder(zi[None]) for zi in z], dim=0)
        return y * scale
