"""Evaluate a private model on private data locally; publish only approved aggregates.

Contract: the full report stays on the owner's machine, outside this repository. The summary is
built from a whitelist, so a field that is not named here cannot reach it. Nothing is ever
committed, pushed or uploaded by this module; publication stays a manual step after approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

from photonic_poc.benchmark import REFERENCE, environment_info, run_benchmark
from photonic_poc.hybrid import HybridOptions
from photonic_poc.inference import BACKEND_NAMES, create_backend
from photonic_poc.models import count_parameters
from photonic_poc.op_profile import profile_model
from photonic_poc.qant_backend import check_supported
from photonic_poc.stability import decision_stability

REPO_ROOT = Path(__file__).resolve().parents[2]
FOLD_BIAS = HybridOptions(fold_bias=True)
BANNER = "NOT APPROVED FOR PUBLICATION"
BACKEND_METRICS = ("f1", "precision", "recall", "roc_auc")
COMPARISON_KEYS = ("delta_f1", "flip_rate", "rel_rms_dev", "delta_ece")
THRESHOLD_KEYS = ("target_precision", "candidate_precision", "fp_rate_change")
TOP_KEYS = frozenset((
    "schema_version", "n_test_approx", "parameters_approx", "total_ops_approx", "offload_fraction",
    "backends", "qant_vs_fp32", "high_precision", "approved_for_publication", "approval",
))  # fmt: skip


def load_model(ref: str) -> nn.Sequential:
    """Import ``package.module:factory`` from the caller's environment and call ``factory()``."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"model reference must look like 'package.module:factory', got {ref!r}")
    if str(Path.cwd()) not in sys.path:  # let a private package in the cwd be found
        sys.path.insert(0, str(Path.cwd()))
    model = getattr(importlib.import_module(module_name), attr)()
    if not isinstance(model, nn.Sequential):
        raise TypeError(f"{ref} returned {type(model).__name__}, expected nn.Sequential")
    return model.eval()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read and validate an .npz: x_test, y_test required; x_val, y_val, kinds_test optional."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    if missing := {"x_test", "y_test"} - set(arrays):
        raise ValueError(f"{path.name}: missing required arrays {sorted(missing)}")
    if ("x_val" in arrays) != ("y_val" in arrays):
        raise ValueError("x_val and y_val must be given together")
    for split in ("test", "val"):
        if f"x_{split}" not in arrays:
            continue
        x, y = arrays[f"x_{split}"], arrays[f"y_{split}"]
        if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
            raise ValueError(f"x_{split} must be a 2-D float array, got {x.dtype} {x.shape}")
        if y.shape != (len(x),):
            raise ValueError(f"y_{split} must have shape ({len(x)},), got {y.shape}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"y_{split} must contain only 0 and 1")
        if not np.isfinite(x).all():
            raise ValueError(f"x_{split} contains NaN or inf")
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x.astype(np.float32), y.astype(np.int64)
    if "x_val" in arrays and arrays["x_val"].shape[1] != arrays["x_test"].shape[1]:
        raise ValueError("x_val and x_test must have the same number of features")
    if "kinds_test" in arrays and arrays["kinds_test"].shape != (len(arrays["x_test"]),):
        raise ValueError("kinds_test must have one label per test row")
    return arrays


def toolkit_available() -> bool:
    return importlib.util.find_spec("qant_native_computing_toolkit") is not None


def evaluate_private(
    model: nn.Sequential, arrays: dict[str, np.ndarray], backends: Sequence[str] | None = None
) -> dict[str, Any]:
    """Full private report. Toolkit backends are skipped, with the reason, when unusable."""
    x, y = arrays["x_test"], arrays["y_test"]
    problems = check_supported(model)
    requested = list(backends) if backends else [REFERENCE, "torch-bf16", "qant"]
    skipped: dict[str, str] = {}
    if "qant" in requested and (problems or not toolkit_available()):
        skipped["qant"] = "unsupported layers: " + "; ".join(problems) if problems else "no toolkit"
    run = [b for b in requested if b not in skipped]
    if REFERENCE not in run:
        run.insert(0, REFERENCE)
    kinds = arrays.get("kinds_test")
    bench = run_benchmark(
        {"private": model},
        x,
        y,
        backends=run,
        kinds=kinds.tolist() if kinds is not None else None,
        latency_repeats=3,
    )
    report: dict[str, Any] = {
        "data": {
            "n_test": len(x),
            "n_negative": int((y == 0).sum()),
            "n_features": x.shape[1],
            "parameters": count_parameters(model),
        },
        "unsupported_layers": problems,
        "skipped_backends": skipped,
        "benchmark": bench,
        "environment": environment_info(),
        "profile": None if problems else profile_model(model),
        "profile_fold_bias": None if problems else profile_model(model, FOLD_BIAS),
    }
    if "qant" in run:
        fp32 = create_backend(REFERENCE, model)
        validation = None
        if "x_val" in arrays:
            validation = (arrays["y_val"], fp32.predict_logits(arrays["x_val"]))
        report["stability"] = decision_stability(
            y,
            fp32.predict_logits(x),
            create_backend("qant", model).predict_logits(x),
            kinds=kinds,
            validation=validation,
        )
    return report


def _sig(value: float, digits: int) -> float:
    return float(f"{value:.{digits}g}")


def aggregate_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Whitelist-built, coarsened summary. Reads named fields only; never copies containers."""
    profile = report["profile"]
    rows = {r["backend"]: r for r in report["benchmark"]["rows"]}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "n_test_approx": int(round(report["data"]["n_test"], -2)),
        "parameters_approx": int(_sig(report["data"]["parameters"], 2)),
        "backends": {
            name: {k: round(float(row["metrics"][k]), 3) for k in BACKEND_METRICS}
            for name, row in rows.items()
        },
    }
    if profile is not None:
        summary["total_ops_approx"] = int(_sig(profile["total_ops"], 2))
        summary["offload_fraction"] = round(float(profile["offload_fraction"]), 2)
    stability = report.get("stability")
    if stability is not None and "qant" in rows:
        ece = stability["calibration_candidate"]["ece"] - stability["calibration_reference"]["ece"]
        summary["qant_vs_fp32"] = {
            "delta_f1": _sig(rows["qant"]["metrics"]["f1"] - rows[REFERENCE]["metrics"]["f1"], 4),
            "flip_rate": _sig(stability["flip_rate"], 4),
            "rel_rms_dev": _sig(stability["rel_rms_dev"], 4),
            "delta_ece": _sig(ece, 4),
        }
        n_neg = max(report["data"]["n_negative"], 1)
        summary["high_precision"] = [
            {
                "target_precision": t["target_precision"],
                "candidate_precision": round(float(t["candidate"]["precision"]), 4),
                "fp_rate_change": _sig(
                    (t["candidate"]["false_positives"] - t["reference"]["false_positives"]) / n_neg,
                    3,
                ),
            }
            for t in stability.get("thresholds", [])
            if t.get("threshold") is not None
        ]
    summary["approved_for_publication"] = False
    summary["approval"] = None
    return summary


def _validate_summary(summary: dict[str, Any]) -> None:
    """Reject any key outside the whitelist, at every level."""

    def check(found: Any, allowed: Sequence[str] | frozenset[str], where: str) -> None:
        extra = set(found) - set(allowed)
        if extra:
            raise ValueError(f"{where}: keys not on the whitelist: {sorted(extra)}")

    check(summary, TOP_KEYS, "summary")
    for name, metrics in summary.get("backends", {}).items():
        if name not in BACKEND_NAMES:
            raise ValueError(f"backends: unknown backend name {name!r}")
        check(metrics, BACKEND_METRICS, f"backends.{name}")
    check(summary.get("qant_vs_fp32", {}), COMPARISON_KEYS, "qant_vs_fp32")
    for row in summary.get("high_precision", []):
        check(row, THRESHOLD_KEYS, "high_precision")


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {BANNER}",
        "",
        "Review every number, then run `approve-summary`.",
        "",
        f"- test pairs (approx.): {summary['n_test_approx']}",
        f"- parameters (approx.): {summary['parameters_approx']}",
        f"- ops per pair (approx.): {summary.get('total_ops_approx', 'n/a')}",
        f"- offload fraction: {summary.get('offload_fraction', 'n/a')}",
        "",
        "| backend | F1 | precision | recall | ROC AUC |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary["backends"].items():
        lines.append(f"| {name} | " + " | ".join(str(m[k]) for k in BACKEND_METRICS) + " |")
    if "qant_vs_fp32" in summary:
        deltas = ", ".join(f"{k}={v}" for k, v in summary["qant_vs_fp32"].items())
        lines += ["", f"qant vs fp32: {deltas}"]
        lines += [
            f"- precision target {r['target_precision']}: candidate precision "
            f"{r['candidate_precision']}, FP rate change {r['fp_rate_change']}"
            for r in summary["high_precision"]
        ]
    return "\n".join(lines) + "\n"


def write_private_outputs(
    report: dict[str, Any], summary: dict[str, Any], out_dir: Path
) -> list[Path]:
    """Write report, summary JSON and summary Markdown. Refuses to write inside this repository."""
    if (REPO_ROOT / ".git").exists() and out_dir.resolve().is_relative_to(REPO_ROOT):
        raise PermissionError(f"refusing to write private output inside the repository: {out_dir}")
    _validate_summary(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "private_report.json": json.dumps(report, indent=2, default=str),
        "aggregate_summary.json": json.dumps(summary, indent=2),
        "aggregate_summary.md": _markdown(summary),
    }
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    return [out_dir / name for name in files]


def approve_summary(summary_path: Path, approved_by: str, out_path: Path) -> dict[str, Any]:
    """Record a human approval of a reviewed summary. Writes ``out_path`` only.

    The sha256 covers the canonical JSON (sorted keys, compact) of the approved summary with
    ``approved_for_publication`` true and the ``approval`` key removed. This function never
    commits, pushes or uploads anything.
    """
    if not approved_by.strip():
        raise ValueError("approved_by must name a person")
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_summary(summary)
    if summary.get("approved_for_publication") or summary.get("approval"):
        raise ValueError("summary is already approved")
    summary["approved_for_publication"] = True
    body = {k: v for k, v in summary.items() if k != "approval"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    summary["approval"] = {
        "approved_by": approved_by.strip(),
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_private_eval(args: argparse.Namespace) -> None:
    report = evaluate_private(load_model(args.model), load_arrays(Path(args.data)), args.backends)
    summary = aggregate_summary(report)
    for path in write_private_outputs(report, summary, Path(args.out)):
        print(path)
    print(f"{BANNER}: review aggregate_summary.md, then run approve-summary.")


def _run_approve(args: argparse.Namespace) -> None:
    approve_summary(Path(args.summary), args.approved_by, Path(args.out))
    print(f"approved summary written to {args.out} (nothing was published)")


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("private-eval", help="evaluate a private model and dataset locally")
    p.add_argument("--model", required=True, help="package.module:factory returning nn.Sequential")
    p.add_argument("--data", required=True, help=".npz: x_test, y_test[, x_val, y_val, kinds_test]")
    p.add_argument("--out", required=True, help="output directory outside this repository")
    p.add_argument("--backends", nargs="+", choices=BACKEND_NAMES, default=None)
    p.set_defaults(func=_run_private_eval)
    a = subparsers.add_parser("approve-summary", help="record human approval of a summary")
    a.add_argument("--summary", required=True)
    a.add_argument("--approved-by", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=_run_approve)
