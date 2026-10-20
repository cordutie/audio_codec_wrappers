"""Metrics of design.md "Metrics": multi-resolution log-mel distance, log-energy envelopes, S1 fidelity report."""
import math

import torch
import torchaudio


def log_mel_distance(x, y, sample_rate: int, n_ffts=(512, 1024, 2048), n_mels: int = 80, eps: float = 1e-5) -> float:
    """Mean |log(Mel x + eps) - log(Mel y + eps)| over channels, bands and frames, averaged over n_ffts. x, y: [C, N]."""
    x, y = torch.as_tensor(x).float(), torch.as_tensor(y).float()
    assert x.shape == y.shape, f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}"
    total = 0.0
    for n_fft in n_ffts:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 4, f_min=0.0, f_max=sample_rate / 2,
            n_mels=n_mels, window_fn=torch.hann_window, power=2.0,
        ).to(x.device)
        d = (torch.log(mel(x) + eps) - torch.log(mel(y) + eps)).abs().mean()
        total += float(d)
    return total / len(n_ffts)


def log_energy_envelope(x, frame: int = 1024, hop: int = 256) -> torch.Tensor:
    """dB, mono mixdown: 10*log10(mean(x^2) + 1e-10) per frame."""
    x = torch.as_tensor(x).float()
    mono = x.mean(dim=0) if x.ndim == 2 else x
    frames = mono.unfold(0, frame, hop)  # [F, frame]
    return 10.0 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else 0.0


def envelope_correlation(x, y, frame: int = 1024, hop: int = 256) -> float:
    """Pearson correlation of the two log-energy envelopes (F3)."""
    return _pearson(log_energy_envelope(x, frame, hop), log_energy_envelope(y, frame, hop))


def is_nonstationary(x, frame: int = 1024, hop: int = 256, min_std_db: float = 3.0) -> bool:
    return bool(log_energy_envelope(x, frame, hop).std() >= min_std_db)


def envelope_lag(x, y, sample_rate: int, frame: int = 1024, hop: int = 256, max_lag_s: float = 0.2) -> int:
    """Lag in samples (positive = y later than x) that maximises the correlation of the log-energy envelopes."""
    ex, ey = log_energy_envelope(x, frame, hop), log_energy_envelope(y, frame, hop)
    n = min(len(ex), len(ey))
    max_lag = int(max_lag_s * sample_rate / hop)
    best_lag, best = 0, -2.0
    for k in range(-max_lag, max_lag + 1):
        # y[t + k] against x[t]
        a = ex[max(0, -k):n - max(0, k)]
        b = ey[max(0, k):n - max(0, -k)]
        if len(a) < 2:
            continue
        c = _pearson(a, b)
        if c > best:
            best, best_lag = c, k
    return best_lag * hop


def _delay(x: torch.Tensor, n: int) -> torch.Tensor:
    """x delayed by n samples with zero fill at the start."""
    y = torch.zeros_like(x)
    if n < x.shape[-1]:
        y[..., n:] = x[..., :x.shape[-1] - n]
    return y


def fidelity_report(x, x_hat, sample_rate: int, codec_info, noise_seed: int = 0) -> dict:
    """F1 (rho_nu), F3 (envelope correlation vs delayed copy) and alignment lag. x, x_hat: [C, N]."""
    x, x_hat = torch.as_tensor(x).float().cpu(), torch.as_tensor(x_hat).float().cpu()
    # RMS-matched Gaussian noise
    g = torch.Generator().manual_seed(noise_seed)
    nu = torch.randn(x.shape, generator=g)
    nu = nu * (x.pow(2).mean().sqrt() / nu.pow(2).mean().sqrt())
    m_hat = log_mel_distance(x, x_hat, sample_rate)
    m_noise = log_mel_distance(x, nu, sample_rate)
    delta_s = 0.25 if codec_info.name == "codicodec" else 0.05
    delayed = _delay(x, int(round(delta_s * sample_rate)))
    return {
        "rho_nu": m_hat / m_noise,
        "M_hat": m_hat,
        "M_noise": m_noise,
        "nonstationary": is_nonstationary(x),
        "c_hat": envelope_correlation(x, x_hat),
        "c_delayed": envelope_correlation(x, delayed),
        "delta_s": delta_s,
        "lag": envelope_lag(x, x_hat, sample_rate),
    }


def s1_thresholds(calibration: dict[str, list[float]]) -> dict[str, float]:
    """tau_k = min(0.9, 1.5 * max rho_nu over the calibration set) per codec (design D18)."""
    return {name: min(0.9, 1.5 * max(rhos)) for name, rhos in calibration.items()}
