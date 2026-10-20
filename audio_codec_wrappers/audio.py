"""Audio utilities: resampling and channel mapping, padding, file loading."""
import math

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def adapt_audio(audio, sample_rate: int, target_rate: int, target_channels: int) -> torch.Tensor:
    """[N], [C_in, N] or [B, C_in, N] -> float32 [B, target_channels, N'] at target_rate."""
    x = torch.as_tensor(audio).float()
    if x.ndim == 1:
        x = x[None, None]
    elif x.ndim == 2:
        x = x[None]
    elif x.ndim != 3:
        raise ValueError(f"audio must have 1, 2 or 3 dims, got shape {tuple(x.shape)}")
    # resample on the last axis
    if sample_rate != target_rate:
        x = torchaudio.functional.resample(x, sample_rate, target_rate)
    # map channels: mono -> stereo duplicates, stereo -> mono is the mean
    c = x.shape[1]
    if c != target_channels:
        if target_channels == 1:
            x = x.mean(dim=1, keepdim=True)
        elif c == 1:
            x = x.expand(-1, target_channels, -1).contiguous()
        else:
            raise ValueError(f"cannot map {c} channels to {target_channels}")
    return x.contiguous()


def pad_to_multiple(audio: torch.Tensor, multiple: int) -> torch.Tensor:
    """Right zero-pad the last axis to a multiple of `multiple`."""
    n = audio.shape[-1]
    extra = math.ceil(n / multiple) * multiple - n
    return F.pad(audio, (0, extra)) if extra else audio


def load_audio(path, start_s: float = 0.0, end_s: float | None = None) -> tuple[torch.Tensor, int]:
    """Read [start_s, end_s) of an audio file; returns ([C, N] float32, sample rate)."""
    sr = sf.info(str(path)).samplerate
    start = int(round(start_s * sr))
    stop = None if end_s is None else int(round(end_s * sr))
    data, _ = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)  # [N, C]
    return torch.from_numpy(data.T.copy()), sr
