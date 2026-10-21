"""Downloads codec weights into the cache and records them in weights.lock.json.

    python scripts/fetch_weights.py --codec spectrostream|codicodec [--route codec|full]

Cache: ~/.cache/audio_codec_wrappers/<codec>/ (override with env ACW_CACHE).
weights.lock.json (repo root) holds, per codec: source repo, revision, route, and for every file the local
path, bytes and sha256.
"""
import argparse
import hashlib
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
LOCK = REPO / "weights.lock.json"

# per codec and route: HF repo, files to fetch
SOURCES = {
    "spectrostream": {
        "repo": "google/magenta-realtime-2",
        # route 1: the codec-only resources (encoder + decoder, 247 MB)
        "codec": ["resources/spectrostream/encoder.safetensors", "resources/spectrostream/decoder.safetensors"],
        # route 2 (fallback): params['soundstream'] of the full model checkpoint (9.8 GB)
        "full": ["checkpoints/mrt2_base.safetensors"],
    },
    # one pickle checkpoint (torch.load with weights_only=False): official HF repo only; the wrapper checks the
    # sha256 recorded here before loading it
    "codicodec": {
        "repo": "SonyCSLParis/codicodec",
        "codec": ["codicodec.pt"],
    },
}


def cache_dir(codec: str) -> pathlib.Path:
    root = pathlib.Path(os.environ.get("ACW_CACHE", pathlib.Path.home() / ".cache" / "audio_codec_wrappers"))
    return root / codec


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(codec: str, route: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    src = SOURCES[codec]
    revision = HfApi().model_info(src["repo"]).sha
    out = cache_dir(codec)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in src[route]:
        path = pathlib.Path(hf_hub_download(src["repo"], name, revision=revision, local_dir=out))
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_of(path)}
        print(f"{codec}: {name} {files[name]['bytes']} bytes sha256 {files[name]['sha256'][:12]}")
    return {"source": f"hf:{src['repo']}", "revision": revision, "route": route, "files": files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codec", required=True, choices=sorted(SOURCES))
    p.add_argument("--route", default="codec", choices=["codec", "full"])
    args = p.parse_args()
    if args.route not in SOURCES[args.codec]:
        p.error(f"{args.codec} has no route {args.route!r}")
    entry = fetch(args.codec, args.route)
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    lock.setdefault(args.codec, {})[args.route] = entry
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print("wrote", LOCK)


if __name__ == "__main__":
    main()
