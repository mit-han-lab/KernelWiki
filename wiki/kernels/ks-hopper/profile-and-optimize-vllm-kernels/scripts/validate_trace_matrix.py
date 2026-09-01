#!/usr/bin/env python3
"""Validate a vLLM torch-profiler capture matrix without modifying the run."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path


DEFAULT_SCENARIOS = ("1k32k", "32k1k", "4k4k", "8k8k")
DEFAULT_STAGES = ("prefill", "decode-early", "decode-mid", "decode-late")


def csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def test_gzip(path: Path) -> str | None:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    except (OSError, EOFError) as exc:
        return str(exc)
    return None


def validate(
    run_root: Path,
    mode: str,
    scenarios: tuple[str, ...],
    stages: tuple[str, ...],
    tp_size: int,
    check_gzip: bool,
) -> dict[str, object]:
    captures = run_root / "profiles" / mode / "captures"
    failures: list[str] = []
    rows: list[dict[str, object]] = []

    campaign_marker = run_root / "profiles" / mode / "campaign.complete"
    if not campaign_marker.is_file():
        failures.append(f"missing campaign marker: {campaign_marker}")

    for scenario in scenarios:
        for stage in stages:
            stage_dir = captures / scenario / stage
            marker = stage_dir / "capture.complete"
            probe = stage_dir / "probe.json"
            traces = sorted(stage_dir.glob("*.pt.trace.json.gz"))
            corrupt: list[dict[str, str]] = []

            if not stage_dir.is_dir():
                failures.append(f"missing stage directory: {stage_dir}")
            if not marker.is_file():
                failures.append(f"missing capture marker: {marker}")
            if not probe.is_file():
                failures.append(f"missing probe metadata: {probe}")
            else:
                try:
                    json.loads(probe.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid probe metadata {probe}: {exc}")

            if len(traces) != tp_size:
                failures.append(
                    f"{scenario}/{stage}: expected {tp_size} traces, found {len(traces)}"
                )

            if check_gzip:
                for trace in traces:
                    error = test_gzip(trace)
                    if error is not None:
                        corrupt.append({"path": str(trace), "error": error})
                        failures.append(f"corrupt gzip trace {trace}: {error}")

            rows.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "directory": str(stage_dir),
                    "capture_complete": marker.is_file(),
                    "probe_json": probe.is_file(),
                    "trace_count": len(traces),
                    "expected_trace_count": tp_size,
                    "corrupt_traces": corrupt,
                }
            )

    expected_stage_count = len(scenarios) * len(stages)
    expected_trace_count = expected_stage_count * tp_size
    observed_trace_count = sum(int(row["trace_count"]) for row in rows)
    return {
        "pass": not failures,
        "run_root": str(run_root),
        "mode": mode,
        "scenarios": list(scenarios),
        "stages": list(stages),
        "tp_size": tp_size,
        "expected_stage_count": expected_stage_count,
        "expected_trace_count": expected_trace_count,
        "observed_trace_count": observed_trace_count,
        "campaign_complete": campaign_marker.is_file(),
        "rows": rows,
        "failures": failures,
    }


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="trace-matrix-self-test-") as tmp:
        root = Path(tmp)
        mode_root = root / "profiles" / "formal"
        (mode_root / "campaign.complete").parent.mkdir(parents=True)
        (mode_root / "campaign.complete").write_text("ok\n", encoding="utf-8")
        for scenario in ("case",):
            for stage in ("prefill", "decode"):
                dest = mode_root / "captures" / scenario / stage
                dest.mkdir(parents=True)
                (dest / "capture.complete").write_text("ok\n", encoding="utf-8")
                (dest / "probe.json").write_text("{}\n", encoding="utf-8")
                for rank in range(2):
                    with gzip.open(dest / f"rank{rank}.pt.trace.json.gz", "wb") as handle:
                        handle.write(b'{"traceEvents": []}\n')
        valid_result = validate(
            root, "formal", ("case",), ("prefill", "decode"), 2, True
        )
        missing_trace = (
            mode_root
            / "captures"
            / "case"
            / "decode"
            / "rank1.pt.trace.json.gz"
        )
        missing_trace.unlink()
        invalid_result = validate(
            root, "formal", ("case",), ("prefill", "decode"), 2, True
        )
        negative_case_detected = (
            not invalid_result["pass"]
            and any(
                "expected 2 traces, found 1" in failure
                for failure in invalid_result["failures"]
            )
        )
        result = {
            "pass": bool(valid_result["pass"] and negative_case_detected),
            "valid_case_pass": valid_result["pass"],
            "valid_case_trace_count": valid_result["observed_trace_count"],
            "negative_case_detected": negative_case_detected,
            "negative_case_failures": invalid_result["failures"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="?")
    parser.add_argument("--mode", default="formal")
    parser.add_argument(
        "--scenarios", type=csv_values, default=DEFAULT_SCENARIOS
    )
    parser.add_argument("--stages", type=csv_values, default=DEFAULT_STAGES)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--skip-gzip-test", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.run_root is None:
        parser.error("run_root is required unless --self-test is used")
    if args.tp_size <= 0:
        parser.error("--tp-size must be positive")

    result = validate(
        args.run_root.resolve(),
        args.mode,
        args.scenarios,
        args.stages,
        args.tp_size,
        not args.skip_gzip_test,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
