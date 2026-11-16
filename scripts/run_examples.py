#!/usr/bin/env python3
"""Run and verify the 3 CPS standard-preset benchmark examples."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


BENCHMARKS = [
    {
        "name": "t30",
        "cluster_seed": 202,
        "ga_seed": 303,
        "makespan": 730.0,
        "icc": 742.0,
        "rtu_pct": 27.37,
    },
    {
        "name": "t30_B",
        "cluster_seed": 42,
        "ga_seed": 404,
        "makespan": 545.0,
        "icc": 1300.0,
        "rtu_pct": 36.17,
    },
    {
        "name": "DenseNet121",
        "cluster_seed": 101,
        "ga_seed": 42,
        "makespan": 97553.0,
        "icc": 2696640.0,
        "rtu_pct": 27.04,
    },
]


def find_default_binary(repo_root: Path) -> Path:
    candidates = [
        repo_root / "build" / "cps_cli.exe",
        repo_root / "build" / "cps_cli",
        repo_root / "build" / "Release" / "cps_cli.exe",
        repo_root / "build" / "Release" / "cps_cli",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate compiled cps_cli binary under ./build. "
        "Please build CPS with CMake first or pass --binary <path>."
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None, help="Path to cps_cli executable")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "build" / "example_runs",
        help="Directory to store example run artifacts",
    )
    args = parser.parse_args()

    binary = args.binary.resolve() if args.binary else find_default_binary(repo_root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Using binary : {binary}")
    print(f"Output root  : {output_root}\n")

    header = (
        f"{'Benchmark':<13} | {'Seeds (C/G)':<11} | {'Makespan':>10} | "
        f"{'ICC':>11} | {'RTU (%)':>8} | {'Time (s)':>8} | {'Status':<6}"
    )
    print(header)
    print("-" * len(header))

    all_passed = True
    for bench in BENCHMARKS:
        name = bench["name"]
        dag_path = repo_root / "examples" / name / "dag.json"
        out_dir = output_root / name

        cmd = [
            str(binary),
            "--dag",
            str(dag_path),
            "--output-dir",
            str(out_dir),
            "--preset",
            "standard",
            "--cluster-seed",
            str(bench["cluster_seed"]),
            "--seed",
            str(bench["ga_seed"]),
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result_file = out_dir / "schedule_result.json"
        if proc.returncode != 0 or not result_file.is_file():
            all_passed = False
            print(f"{name:<13} | {bench['cluster_seed']}/{bench['ga_seed']:<7} | FAILED (rc={proc.returncode})")
            if proc.stderr:
                print(proc.stderr.strip(), file=sys.stderr)
            continue

        data = json.loads(result_file.read_text(encoding="utf-8"))
        summary = data["summary"]
        runtime_s = data["runtime_seconds"]["total"]

        ms = float(summary["makespan"])
        icc = float(summary["icc"])
        rtu_pct = float(summary["rtu_pct"])

        ok_ms = math.isclose(ms, bench["makespan"], abs_tol=1e-4)
        ok_icc = math.isclose(icc, bench["icc"], abs_tol=1e-4)
        ok_rtu = math.isclose(round(rtu_pct, 2), bench["rtu_pct"], abs_tol=1e-2)
        passed = bool(summary["feasible"]) and bool(summary["layout_valid"]) and ok_ms and ok_icc and ok_rtu
        if not passed:
            all_passed = False

        seeds_str = f"{bench['cluster_seed']}/{bench['ga_seed']}"
        status = "PASS" if passed else "DIFF"
        print(
            f"{name:<13} | {seeds_str:<11} | {ms:>10.1f} | "
            f"{icc:>11.1f} | {rtu_pct:>7.2f}% | {runtime_s:>7.2f}s | {status:<6}"
        )
        if not passed:
            print(
                f"  Expected: Makespan={bench['makespan']:.1f}, "
                f"ICC={bench['icc']:.1f}, RTU={bench['rtu_pct']:.2f}%",
                file=sys.stderr,
            )

    print("-" * len(header))
    if all_passed:
        print("All 3 standard-preset benchmark examples reproduced exact paper metrics.")
        return 0
    print("One or more benchmark examples diverged from expected metrics.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
