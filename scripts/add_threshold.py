"""Append one codec's calibration entry to the wrappers' s1_thresholds.json.

Usage: python scripts/add_threshold.py <calibration_json> <codec> [--target s1_thresholds.json]
Takes calibration_rho_nu[codec] from the calibration json, applies the frozen rule
tau = min(0.9, 1.5 * max rho_nu over clips and seeds) and refuses to change an existing key.
"""
import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("codec")
    ap.add_argument("--target", default=str(pathlib.Path(__file__).resolve().parents[1] / "s1_thresholds.json"))
    args = ap.parse_args()

    calib = json.loads(pathlib.Path(args.calibration).read_text())
    target_path = pathlib.Path(args.target)
    target = json.loads(target_path.read_text())

    # refuse to overwrite an existing entry
    if args.codec in target["tau"] or args.codec in target["calibration_rho_nu"]:
        raise SystemExit(f"{args.codec} already in {target_path}; refusing to change it")

    rho = calib["calibration_rho_nu"][args.codec]
    rhos = [r for clip in rho.values() for r in clip.values()]
    tau = min(0.9, 1.5 * max(rhos))

    # the calibration set must be the frozen one
    assert calib["calibration_clips"] == target["calibration_clips"], "calibration clips differ"
    assert calib["seeds"] == target["seeds"], "seeds differ"

    target["tau"][args.codec] = tau
    target["calibration_rho_nu"][args.codec] = rho
    target_path.write_text(json.dumps(target, indent=2))
    print(f"{args.codec}: tau = {tau} from {len(rhos)} values (max rho_nu {max(rhos)})")


if __name__ == "__main__":
    main()
