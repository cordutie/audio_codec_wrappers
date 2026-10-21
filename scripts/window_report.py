"""Window versus stream decoding of the causal codecs, and the empirical decoder memory R (design D-f, E1b; plan slice A6).

    python scripts/window_report.py --clips samples/guitar.wav ../datasets/microtex/raw/rain.wav:20:30 ... \
        --codecs encodec24k spectrostream --window-seconds 0 0.05 0.1 0.2 0.4 0.8 1.6 3.2 --block-seconds 0.1 \
        --out ../runs/window_decode

For every codec, clip and window N_w (ceiled to frames) the clip's latents are decoded block by block with
`window_decoder(N_w)` and with `streaming_decoder()`. Writes window.json, window.png and env.json. No thresholds:
the report only states the smallest window reaching 1e-4 max abs error and 60 dB against the stream.
"""
import argparse
import json
import math
import os
import pathlib
import platform
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from audio_codec_wrappers import load_codec  # noqa: E402
from audio_codec_wrappers.audio import load_audio  # noqa: E402

DEFAULT_WINDOW_SECONDS = 1.6  # the AR past window of SCAPES (design)
TOL_ABS, TOL_SNR = 1e-4, 60.0


def parse_clip(spec: str) -> tuple[pathlib.Path, float, float | None, str]:
    parts = spec.split(":")
    path = pathlib.Path(parts[0])
    start = float(parts[1]) if len(parts) > 1 else 0.0
    end = float(parts[2]) if len(parts) > 2 else None
    name = path.stem if len(parts) == 1 else f"{path.stem}_{parts[1]}_{parts[2]}"
    return path, start, end, name


def sync(codec) -> None:
    torch.cuda.synchronize()
    if codec.info.backend == "jax":
        import jax
        (jax.device_put(0.0) + 0).block_until_ready()


def run_blocks(dec, z: torch.Tensor, block: int, codec) -> tuple[torch.Tensor, list[float]]:
    """Pushes z in blocks of `block` frames, then flushes; returns the output and the seconds per push."""
    dec.reset()
    outs, times = [], []
    for t in range(0, z.shape[-1], block):
        sync(codec)
        t0 = time.perf_counter()
        outs.append(dec.push(z[..., t:t + block]))
        sync(codec)
        times.append(time.perf_counter() - t0)
    outs.append(dec.flush())
    return torch.cat(outs, dim=-1), times


def snr_db(ref: torch.Tensor, err: torch.Tensor) -> float:
    num, den = float(ref.double().pow(2).sum()), float(err.double().pow(2).sum())
    return float("inf") if den == 0 else 10.0 * math.log10(max(num, 1e-30) / den)


def decoder_memory(codec, z: torch.Tensor) -> dict:
    """Perturb frame j = T/2 with N(0,1) noise, stream-decode, last output sample that changes by more than 1e-6 (R2-C6)."""
    info, h = codec.info, codec.info.hop_length
    T = z.shape[-1]
    j = T // 2
    g = torch.Generator().manual_seed(0)
    z2 = z.clone()
    z2[..., j] += torch.randn(z2[..., j].shape, generator=g).to(z2.device)
    dec = codec.streaming_decoder(batch_size=1)
    L = dec.latency_samples
    a, _ = run_blocks(dec, z, T, codec)
    b, _ = run_blocks(dec, z2, T, codec)
    diff = ((a - b).abs() > 1e-6).any(dim=1)[0, L:L + T * h]
    changed = torch.nonzero(diff).flatten()
    if changed.numel() == 0:
        return {"j": j, "T": T, "R_frames": None, "note": "no change above 1e-6"}
    first, last = int(changed.min()), int(changed.max())
    never_ends = last >= T * h - 1
    R = math.ceil((last + 1) / h) - j
    return {"j": j, "T": T, "first_changed_sample": first, "last_changed_sample": last,
            "R_frames": R, "R_seconds": R * h / info.sample_rate,
            "reaches_clip_end": never_ends,
            "R_report": f">= {T - j} frames (clip length after frame j)" if never_ends else f"{R} frames",
            "after_frame_j_frames": T - j}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--codecs", nargs="+", default=["encodec24k", "spectrostream"])
    ap.add_argument("--window-seconds", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    ap.add_argument("--block-seconds", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [parse_clip(c) for c in args.clips]
    results: dict = {}
    memory: dict = {}
    for name in args.codecs:
        codec = load_codec(name, device="cuda")
        info = codec.info
        block = math.ceil(args.block_seconds * info.frame_rate - 1e-9)
        windows = [math.ceil(w * info.frame_rate - 1e-9) for w in args.window_seconds]
        results[name] = {"block_frames": block, "frame_rate": info.frame_rate, "latency_samples": None, "clips": {}}
        memory[name] = {}
        for path, start, end, clip in clips:
            x, sr = load_audio(path, start, end)
            z = codec.encode(x, sr).z
            T = z.shape[-1]
            sdec = codec.streaming_decoder(batch_size=1)
            L = sdec.latency_samples
            results[name]["latency_samples"] = L
            ref, t_stream = run_blocks(sdec, z, block, codec)
            ref = ref[..., L:L + T * info.hop_length]
            entries = {}
            for w_s, w in zip(args.window_seconds, windows):
                out_w, t_w = run_blocks(codec.window_decoder(w, batch_size=1), z, block, codec)
                out_w = out_w[..., L:L + T * info.hop_length]
                err = (out_w.double() - ref.double())
                entries[str(w_s)] = {
                    "window_frames": w, "window_seconds_ceiled": w / info.frame_rate,
                    "max_abs": float(err.abs().max()), "snr_db": snr_db(ref, err),
                    "decode_seconds_per_block_median": statistics.median(t_w),
                    "decode_seconds_per_block_mean": statistics.fmean(t_w),
                    "stream_seconds_per_block_median": statistics.median(t_stream),
                }
                print(f"[{name}] {clip} window {w_s} s ({w} frames): max_abs {entries[str(w_s)]['max_abs']:.2e} "
                      f"snr {entries[str(w_s)]['snr_db']:.1f} dB, {entries[str(w_s)]['decode_seconds_per_block_median'] * 1e3:.1f} ms/block",
                      flush=True)
            results[name]["clips"][clip] = entries
            memory[name][clip] = decoder_memory(codec, z)
            print(f"[{name}] {clip} decoder memory:", memory[name][clip].get("R_report"), flush=True)
        # smallest window reaching both tolerances: per clip and on all clips
        ok_all = {}
        for w_s in args.window_seconds:
            ok_all[w_s] = all(e[str(w_s)]["max_abs"] <= TOL_ABS and e[str(w_s)]["snr_db"] >= TOL_SNR
                              for e in results[name]["clips"].values())
        good = [w for w in args.window_seconds if ok_all[w]]
        results[name]["smallest_window_all_clips"] = (
            {"seconds_requested": min(good), "frames": math.ceil(min(good) * info.frame_rate - 1e-9)} if good else None)
        results[name]["smallest_window_per_clip"] = {}
        for clip, e in results[name]["clips"].items():
            g = [w for w in args.window_seconds if e[str(w)]["max_abs"] <= TOL_ABS and e[str(w)]["snr_db"] >= TOL_SNR]
            results[name]["smallest_window_per_clip"][clip] = (
                {"seconds_requested": min(g), "frames": math.ceil(min(g) * info.frame_rate - 1e-9)} if g else None)
        default_frames = math.ceil(DEFAULT_WINDOW_SECONDS * info.frame_rate - 1e-9)
        rs = [m["R_frames"] for m in memory[name].values() if m["R_frames"] is not None]
        reaches = any(m.get("reaches_clip_end") for m in memory[name].values())
        look = info.decoder_lookahead_frames
        results[name]["decoder_memory"] = {
            "per_clip": memory[name], "default_window_frames": default_frames,
            "R_max_frames": max(rs) if rs else None, "any_clip_reaches_end": reaches,
            "default_covers_R": (None if reaches or not rs else default_frames >= max(rs)),
            "default_covers_R_plus_lookahead_plus_1": (None if reaches or not rs else default_frames >= max(rs) + look + 1),
        }
        del codec
        torch.cuda.empty_cache()

    (out / "window.json").write_text(json.dumps(results, indent=2))
    env = {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "XLA_FLAGS": os.environ.get("XLA_FLAGS"), "command": " ".join(sys.argv)}
    (out / "env.json").write_text(json.dumps(env, indent=2))

    # figure: max abs error against window length, one panel per codec, one colour per clip
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(args.codecs), figsize=(5.5 * len(args.codecs), 4), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], args.codecs):
        for ci, (clip, e) in enumerate(results[name]["clips"].items()):
            ws = [e[str(w)]["window_seconds_ceiled"] for w in args.window_seconds]
            ax.semilogy(ws, [max(e[str(w)]["max_abs"], 1e-12) for w in args.window_seconds], "o-", color=cm(ci), label=clip)
        ax.axhline(TOL_ABS, color="k", ls="--", lw=1, label="1e-4")
        ax.set_title(name)
        ax.set_xlabel("window (s, ceiled to frames)")
    axes[0][0].set_ylabel("max |window - stream|")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "window.png", dpi=130)
    print(json.dumps({n: {k: v for k, v in r.items() if k not in ("clips",)} for n, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
