"""S1 build report: round trips of the real codecs on a set of clips (design S1, D18; plan slice A6).

    python scripts/roundtrip_report.py --clips samples/guitar.wav ../datasets/microtex/raw/rain.wav:20:30 \
        --codecs all --seeds 0 1 2 --calibrate --out ../runs/s1_build
    python scripts/roundtrip_report.py --clips ... --codecs all --thresholds s1_thresholds.json --out ...

A clip is `path` or `path:start_s:end_s`. For every clip and codec it writes <out>/<codec>/<clip>/report.json
(fidelity report, length check, streaming errors, latency, timing, memory), figures, and env.json.
--calibrate also writes s1_thresholds.json; --thresholds adds a pass/fail per check.
"""
import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from audio_codec_wrappers import list_codecs, load_codec  # noqa: E402
from audio_codec_wrappers.audio import load_audio  # noqa: E402
from audio_codec_wrappers.metrics import fidelity_report, s1_thresholds  # noqa: E402

REAL_CODECS = ["encodec24k", "spectrostream", "encodec48k"]
F3_TIE = 0.005  # tie tolerance of the locked test (Esteban, 2026-10-21)
LAG_MAX = 256  # samples, one envelope hop
STREAM_TOL = 1e-4  # GPU streaming tolerance (design S1 (v))


# ---------- helpers ----------

def parse_clip(spec: str) -> tuple[pathlib.Path, float, float | None, str]:
    """`path[:start:end]` -> (path, start_s, end_s, name)."""
    parts = spec.split(":")
    path = pathlib.Path(parts[0])
    start = float(parts[1]) if len(parts) > 1 else 0.0
    end = float(parts[2]) if len(parts) > 2 else None
    name = path.stem if len(parts) == 1 else f"{path.stem}_{parts[1]}_{parts[2]}"
    return path, start, end, name


def chunk_sizes(kind: str, total: int) -> list[int]:
    """Chunkings of the streaming check: all-1, all-7, random sizes 1..13 (seed 0)."""
    if kind == "random":
        g = torch.Generator().manual_seed(0)
        sizes, s = [], 0
        while s < total:
            n = min(int(torch.randint(1, 14, (1,), generator=g)), total - s)
            sizes.append(n)
            s += n
        return sizes
    n = int(kind)
    return [min(n, total - i) for i in range(0, total, n)]


def stream(dec, z: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    dec.reset()
    outs, t = [], 0
    for n in sizes:
        outs.append(dec.push(z[..., t:t + n]))
        t += n
    outs.append(dec.flush())
    return torch.cat(outs, dim=-1)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double().cpu() - b.double().cpu()).abs().max())


def sync(codec) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if codec.info.backend == "jax":
        import jax
        (jax.device_put(0.0) + 0).block_until_ready()


def jax_peak_bytes() -> int | None:
    import jax
    stats = jax.local_devices()[0].memory_stats()
    return None if not stats else int(stats.get("peak_bytes_in_use", 0))


def timed(codec, fn):
    """Runs fn once untimed (compile / warm-up), then once timed; returns (result, seconds)."""
    fn()
    sync(codec)
    t0 = time.perf_counter()
    out = fn()
    sync(codec)
    return out, time.perf_counter() - t0


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "XLA_FLAGS": __import__("os").environ.get("XLA_FLAGS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": __import__("os").environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_matmul_precision": "highest (jax.default_matmul_precision decorator inside the spectrostream wrapper)",
    }
    try:
        import jax
        env["jax"] = jax.__version__
        env["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as e:  # jax is optional for this field
        env["jax"] = f"unavailable: {e}"
    env["git_sha"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env["git_dirty"] = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                           capture_output=True, text=True).stdout.strip())
    env["weights_lock"] = json.loads((REPO / "weights.lock.json").read_text()) if (REPO / "weights.lock.json").exists() else None
    env["command"] = " ".join(sys.argv)
    return env


# ---------- one (codec, clip) ----------

def run_one(codec, x: torch.Tensor, sr: int, seeds: list[int]) -> dict:
    info = codec.info
    out: dict = {"codec": info.name, "latency_samples": 0, "seeds": {}}
    xp = codec.prepare(x, sr)
    n = xp.shape[-1]
    out["num_samples"] = n
    out["duration_s"] = n / info.sample_rate

    # encode (warm-up + timed), peak GPU memory of the torch allocator
    torch.cuda.reset_peak_memory_stats()
    enc, t_enc = timed(codec, lambda: codec.encode(x, sr))
    out["encode_seconds"] = t_enc
    z, scale = enc.z, enc.scale
    out["latent_shape"] = list(z.shape)
    out["length_check"] = {"frames": z.shape[-1], "expected_frames": info.num_frames(n)}
    out["length_check"]["ok"] = z.shape[-1] == info.num_frames(n)

    # decode per seed (deterministic codecs: one run)
    ys = {}
    for seed in (seeds if info.stochastic_decoder else [0]):
        torch.cuda.reset_peak_memory_stats()
        y, t_dec = timed(codec, lambda: codec.decode(z, scale, seed=seed))
        ys[seed] = y
        rep = fidelity_report(xp[0].cpu(), y[0, :, :n].cpu(), info.sample_rate, info, noise_seed=0)
        rep["decode_seconds"] = t_dec
        rep["decode_length_ok"] = tuple(y.shape) == (1, info.channels, z.shape[-1] * info.hop_length)
        rep["peak_gpu_bytes_torch"] = int(torch.cuda.max_memory_allocated())
        out["seeds"][str(seed)] = rep
    out["peak_gpu_bytes_jax_process"] = jax_peak_bytes() if info.backend == "jax" else None

    # streaming (causal codecs): chunkings against the one-shot push, and against decode
    if info.causal:
        dec = codec.streaming_decoder(batch_size=1)
        L = dec.latency_samples
        out["latency_samples"] = L
        out["latency_seconds"] = L / info.sample_rate
        T = z.shape[-1]
        ref = stream(dec, z, [T])
        full = ys[0]
        errs = {"vs_decode_oneshot": max_abs(ref[..., L:L + T * info.hop_length], full)}
        for kind in ("1", "7", "random"):
            o = stream(dec, z, chunk_sizes(kind, T))
            errs[f"chunk_{kind}_vs_oneshot"] = max_abs(o, ref)
            errs[f"chunk_{kind}_vs_decode"] = max_abs(o[..., L:L + T * info.hop_length], full)
        out["streaming_max_abs"] = errs

    # EnCodec 24k: wrapper decode against the official decoder, per 0.1 s
    if info.name == "encodec24k":
        with torch.no_grad():
            d = (ys[0] - codec.decode_official(z)).abs().amax(dim=(0, 1)).cpu()
        w = int(0.1 * info.sample_rate)
        per = [float(d[i:i + w].max()) for i in range(0, d.numel(), w)]
        out["decode_vs_official"] = {"window_s": 0.1, "max_abs": per}
    return out


def check_thresholds(rep: dict, tau: dict | None, info) -> dict:
    """Pass/fail per check for every seed: F1, F2, F3, lag, length, streaming."""
    checks = {}
    for seed, r in rep["seeds"].items():
        c = {"F1": r["rho_nu"] < 1.0, "lag": abs(r["lag"]) <= LAG_MAX, "length": r["decode_length_ok"] and rep["length_check"]["ok"]}
        if tau is not None:
            c["F2"] = r["rho_nu"] <= tau[info.name]
        # F3 only on non-stationary clips; strict margin reported next to the pass
        c["F3"] = (r["c_hat"] >= r["c_delayed"] - F3_TIE) if r["nonstationary"] else None
        c["F3_strict"] = (r["c_hat"] >= r["c_delayed"]) if r["nonstationary"] else None
        checks[seed] = c
    if "streaming_max_abs" in rep:
        checks["streaming"] = {k: v <= STREAM_TOL for k, v in rep["streaming_max_abs"].items()}
    return checks


# ---------- figures ----------

def make_figures(results: dict, tau: dict | None, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codecs = list(results)
    clips = list(next(iter(results.values())))
    colours = plt.get_cmap("tab10")
    # rho_nu per clip and codec
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ci, clip in enumerate(clips):
        for k, codec in enumerate(codecs):
            rhos = [r["rho_nu"] for r in results[codec][clip]["seeds"].values()]
            ax.scatter([k + 0.12 * (ci - len(clips) / 2)] * len(rhos), rhos, color=colours(ci), s=28,
                       label=clip if k == 0 else None)
    ax.axhline(1.0, color="k", lw=1, label="1 (noise)")
    if tau is not None:
        for k, codec in enumerate(codecs):
            ax.hlines(tau[codec], k - 0.4, k + 0.4, colors="r", lw=2, label="tau" if k == 0 else None)
    ax.set_xticks(range(len(codecs)), codecs)
    ax.set_ylabel("rho_nu")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rho_nu.png", dpi=130)
    plt.close(fig)
    # envelope correlations: reconstruction against delayed copy, one panel per codec
    fig, axes = plt.subplots(1, len(codecs), figsize=(3.2 * len(codecs), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, codec in zip(axes, codecs):
        for ci, clip in enumerate(clips):
            for r in results[codec][clip]["seeds"].values():
                if r["nonstationary"]:
                    ax.scatter(r["c_delayed"], r["c_hat"], color=colours(ci), s=28, label=clip)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(codec, fontsize=9)
        ax.set_xlabel("c(x, delayed x)")
    axes[0].set_ylabel("c(x, x_hat)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "envelope_correlation.png", dpi=130)
    plt.close(fig)
    # EnCodec 24k decode - decode_official against time
    if "encodec24k" in results:
        fig, ax = plt.subplots(figsize=(7, 4))
        for ci, clip in enumerate(clips):
            per = results["encodec24k"][clip]["decode_vs_official"]["max_abs"]
            ax.semilogy([0.1 * (i + 1) for i in range(len(per))], per, color=colours(ci), label=clip)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("max |decode - decode_official| per 0.1 s")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "encodec24k_decode_vs_official.png", dpi=130)
        plt.close(fig)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--codecs", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--thresholds", default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = REAL_CODECS if args.codecs == ["all"] else args.codecs
    assert all(n in list_codecs() for n in names), names
    clips = [parse_clip(c) for c in args.clips]
    tau_file = json.loads(pathlib.Path(args.thresholds).read_text())["tau"] if args.thresholds else None

    (out / "env.json").write_text(json.dumps(environment(), indent=2))
    results: dict = {}
    for name in names:
        codec = load_codec(name, device="cuda")
        results[name] = {}
        for path, start, end, clip in clips:
            x, sr = load_audio(path, start, end)
            print(f"[{name}] {clip}: {x.shape[-1] / sr:.2f} s at {sr} Hz", flush=True)
            rep = run_one(codec, x, sr, args.seeds)
            rep["clip"] = {"path": str(path), "start_s": start, "end_s": end}
            if tau_file is not None:
                rep["checks"] = check_thresholds(rep, tau_file, codec.info)
            else:
                rep["checks"] = check_thresholds(rep, None, codec.info)
            d = out / name / clip
            d.mkdir(parents=True, exist_ok=True)
            (d / "report.json").write_text(json.dumps(rep, indent=2))
            results[name][clip] = rep
            print("   rho_nu", {s: round(r["rho_nu"], 4) for s, r in rep["seeds"].items()},
                  "c_hat/c_delayed", [(round(r["c_hat"], 4), round(r["c_delayed"], 4)) for r in rep["seeds"].values()][:1],
                  "lag", [r["lag"] for r in rep["seeds"].values()],
                  "stream", {k: f"{v:.1e}" for k, v in rep.get("streaming_max_abs", {}).items() if k.startswith("vs") or "random" in k},
                  flush=True)
        del codec
        torch.cuda.empty_cache()

    # F1 on every clip; stop-and-report if a codec is worse than noise
    f1_fail = [(n, c, s) for n in results for c, r in results[n].items() for s, v in r["seeds"].items() if v["rho_nu"] >= 1.0]
    summary: dict = {"F1_failures": f1_fail}
    tau = None
    if args.calibrate:
        calib = {n: [v["rho_nu"] for r in results[n].values() for v in r["seeds"].values()] for n in results}
        tau = s1_thresholds(calib)
        thr = {"tau": tau, "rule": "min(0.9, 1.5 * max over K of rho_nu)",
               "calibration_rho_nu": {n: {c: {s: v["rho_nu"] for s, v in r["seeds"].items()} for c, r in results[n].items()} for n in results},
               "calibration_clips": args.clips, "seeds": args.seeds}
        (out / "s1_thresholds.json").write_text(json.dumps(thr, indent=2))
        summary["tau"] = tau
    elif tau_file is not None:
        tau = tau_file
    if tau is not None:
        # F2 table
        summary["F2_failures"] = [(n, c, s) for n in results for c, r in results[n].items()
                                  for s, v in r["seeds"].items() if v["rho_nu"] > tau[n]]
    summary["F3_failures"] = [(n, c, s) for n in results for c, r in results[n].items()
                              for s, ch in r["checks"].items() if s.isdigit() and ch["F3"] is False]
    summary["F3_strict_failures"] = [(n, c, s) for n in results for c, r in results[n].items()
                                     for s, ch in r["checks"].items() if s.isdigit() and ch["F3_strict"] is False]
    summary["lag_failures"] = [(n, c, s) for n in results for c, r in results[n].items()
                               for s, ch in r["checks"].items() if s.isdigit() and not ch["lag"]]
    summary["streaming_failures"] = [(n, c, k) for n in results for c, r in results[n].items()
                                     for k, v in r["checks"].get("streaming", {}).items() if not v]
    summary["length_failures"] = [(n, c, s) for n in results for c, r in results[n].items()
                                  for s, ch in r["checks"].items() if s.isdigit() and not ch["length"]]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(results, tau, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
