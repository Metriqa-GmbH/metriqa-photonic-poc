"""Experiment B: how do simulator numerics change decisions, and where?

H_B1: the toolkit simulator flips < 0.5% of decisions for every studied model, and only pairs
whose float32 margin is below the simulator's 99.9th-percentile logit deviation can flip.
H_B2: ambiguous pair kinds flip more often only because they lie closer to the boundary
(observed flips per kind ≈ flips expected from their margins).
H_B3: calibration (ECE) changes by less than 0.005, and precision at high-precision
thresholds tuned in float32 changes by less than 0.002.
H_B4: logit deviation grows with depth, faster for KAN than for MLP.
Hardware-only effects (analog noise, cosine shape) are probed with explicitly assumed models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from photonic_poc.datasets import Dataset
from photonic_poc.experiments.config import (
    dataset_from,
    mean_std,
    select_learning_rates,
    train_seeds,
    write_json,
)
from photonic_poc.hybrid import HybridExecutor, HybridOptions
from photonic_poc.inference import NoisyTorchBackend, TorchBackend
from photonic_poc.metrics import classification_metrics, output_agreement
from photonic_poc.models import ArchSpec, FourierKAN
from photonic_poc.probes import boundary_inputs, distorted_cos, uniform_inputs
from photonic_poc.stability import decision_stability, margin_profile
from photonic_poc.stage import FP32
from photonic_poc.sweep import Trained

POLICIES = {
    "toolkit": HybridOptions(),
    "host-fp32": HybridOptions(host_precision=FP32),
    "fold-bias": HybridOptions(fold_bias=True),
    "kan-native": HybridOptions(kan_impl="native", host_precision=FP32),
    "kan-native+wrap": HybridOptions(kan_impl="native", host_precision=FP32, wrap_phases=True),
    "kan-factorized": HybridOptions(kan_impl="factorized"),
}


def _study_models(arch: dict[str, Any]) -> list[str]:
    names = ["logreg"]
    for picks in arch["champions"].values():
        names += [picks.get("best", ""), picks.get("compact", "")]
    names += ["mlp-32x16", "kan-4-g4"]
    known = arch["training"]["selected_learning_rates"]
    return [n for n in dict.fromkeys(names) if n in known]


def _f1(y: np.ndarray, logits: np.ndarray) -> float:
    return classification_metrics(y, logits)["f1"]


def _seed_report(run: Trained, data: Dataset, rcfg: dict[str, Any], seed: int) -> dict[str, Any]:
    model, family = run.model, ArchSpec.parse(run.job.arch).family
    fp32 = TorchBackend(model).predict_logits
    ref, ref_val = fp32(data.x_test), fp32(data.x_val)
    policies = [p for p in POLICIES if family == "kan" or not p.startswith("kan")]
    logits = {p: HybridExecutor(model, POLICIES[p]).predict_logits(data.x_test) for p in policies}
    logits["torch-bf16"] = TorchBackend(model, torch.bfloat16).predict_logits(data.x_test)
    toolkit = HybridExecutor(model).predict_logits
    report: dict[str, Any] = {
        "stability": decision_stability(
            data.y_test,
            ref,
            logits["toolkit"],
            data.kinds_test,
            (data.y_val, ref_val),
            rcfg["precision_targets"],
        ),
        "policies": {
            p: output_agreement(ref, v) | {"delta_f1": _f1(data.y_test, v) - _f1(data.y_test, ref)}
            for p, v in logits.items()
        },
        "f1_fp32": _f1(data.y_test, ref),
    }
    probes = {
        "test": data.x_test,
        "uniform": uniform_inputs(rcfg["uniform_samples"], data.n_features, seed),
        "boundary": boundary_inputs(fp32, data.x_test, rcfg["boundary_samples"], seed),
    }
    report["distributions"] = {}
    for name, x in probes.items():
        r, c = fp32(x), toolkit(x)
        report["distributions"][name] = output_agreement(r, c) | {
            "p999_abs_dev": float(np.quantile(np.abs(c - r), 0.999)),
            "margin_profile": margin_profile(r, c),
        }
    report["noise"] = []
    for std in rcfg["noise_std"]:
        reps = [
            NoisyTorchBackend(model, std, s).predict_logits(data.x_test)
            for s in range(rcfg["noise_seeds"])
        ]
        report["noise"].append(
            {
                "std": std,
                "flip_rate": float(np.mean([np.mean((v > 0) != (ref > 0)) for v in reps])),
                "delta_f1": float(np.mean([_f1(data.y_test, v) for v in reps])) - report["f1_fp32"],
            }
        )
    report["tcos"] = []
    for eps in rcfg["tcos_distortion"]:
        v = TorchBackend(model, periodic=distorted_cos(eps)).predict_logits(data.x_test)
        report["tcos"].append(
            {
                "epsilon": eps,
                "flip_rate": float(np.mean((v > 0) != (ref > 0))),
                "delta_f1": _f1(data.y_test, v) - report["f1_fp32"],
            }
        )
    phases = [m.phase.detach().numpy() for m in model.modules() if isinstance(m, FourierKAN)]
    if phases:
        flat = np.concatenate([p.ravel() for p in phases])
        report["phase_stats"] = {
            "max_abs": float(np.abs(flat).max()),
            "share_outside_pi": float(np.mean(np.abs(flat) > np.pi)),
        }
    return report


def _pool(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Seed-level reports -> mean/std of scalars, pooled counts for profiles."""
    st = [r["stability"] for r in reports]
    out: dict[str, Any] = {
        "f1_fp32": mean_std([r["f1_fp32"] for r in reports]),
        "flip_rate": mean_std([s["flip_rate"] for s in st]),
        "flips_total": int(sum(s["flips"] for s in st)),
        "rel_rms_dev": mean_std([s["rel_rms_dev"] for s in st]),
        "p999_abs_dev": mean_std([s["p999_abs_dev"] for s in st]),
        "delta_ece": mean_std(
            [s["calibration_candidate"]["ece"] - s["calibration_reference"]["ece"] for s in st]
        ),
        "delta_brier": mean_std(
            [s["calibration_candidate"]["brier"] - s["calibration_reference"]["brier"] for s in st]
        ),
        "ece_fp32": mean_std([s["calibration_reference"]["ece"] for s in st]),
        "reliability_fp32": st[0]["calibration_reference"]["bins"],
        "reliability_toolkit": st[0]["calibration_candidate"]["bins"],
        "deviation_by_magnitude": st[0]["deviation_by_magnitude"],
    }
    out["margin_profile"] = _sum_profiles([s["margin_profile"] for s in st])
    kinds: dict[str, dict[str, float]] = {}
    for s in st:
        for k in s["kinds"]:
            agg = kinds.setdefault(k["kind"], {"n": 0, "observed_flips": 0, "expected_flips": 0.0})
            for key in agg:
                agg[key] += k[key]
    out["kinds"] = kinds
    out["thresholds"] = []
    for i, first in enumerate(st[0]["thresholds"]):
        rows = [s["thresholds"][i] for s in st if s["thresholds"][i]["threshold"] is not None]
        if not rows:
            out["thresholds"].append(
                {"target_precision": first["target_precision"], "available": False}
            )
            continue
        out["thresholds"].append(
            {
                "target_precision": first["target_precision"],
                "precision_fp32": mean_std([r["reference"]["precision"] for r in rows]),
                "precision_toolkit": mean_std([r["candidate"]["precision"] for r in rows]),
                "recall_fp32": mean_std([r["reference"]["recall"] for r in rows]),
                "fp_fp32": int(sum(r["reference"]["false_positives"] for r in rows)),
                "fp_toolkit": int(sum(r["candidate"]["false_positives"] for r in rows)),
                "flips_at_threshold": int(sum(r["flips_at_threshold"] for r in rows)),
                "seeds": len(rows),
            }
        )
    out["policies"] = {
        p: {
            k: mean_std([r["policies"][p][k] for r in reports])
            for k in ("rel_rms_diff", "decision_flips", "delta_f1")
        }
        for p in reports[0]["policies"]
    }
    out["distributions"] = {
        name: {
            "rel_rms_diff": mean_std([r["distributions"][name]["rel_rms_diff"] for r in reports]),
            "p999_abs_dev": mean_std([r["distributions"][name]["p999_abs_dev"] for r in reports]),
            "flip_rate": mean_std(
                [r["distributions"][name]["decision_flip_rate"] for r in reports]
            ),
            "margin_profile": _sum_profiles(
                [r["distributions"][name]["margin_profile"] for r in reports]
            ),
        }
        for name in reports[0]["distributions"]
    }
    for key, axis in (("noise", "std"), ("tcos", "epsilon")):
        out[key] = [
            {
                axis: row[axis],
                "flip_rate": mean_std([r[key][i]["flip_rate"] for r in reports]),
                "delta_f1": mean_std([r[key][i]["delta_f1"] for r in reports]),
            }
            for i, row in enumerate(reports[0][key])
        ]
    if "phase_stats" in reports[0]:
        out["phase_stats"] = {
            k: max(r["phase_stats"][k] for r in reports) for k in reports[0]["phase_stats"]
        }
    return out


def _sum_profiles(profiles: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    rows = []
    for i, first in enumerate(profiles[0]):
        n = sum(p[i]["n"] for p in profiles)
        flips = sum(p[i]["flips"] for p in profiles)
        rows.append(
            {
                "lo": first["lo"],
                "hi": first["hi"],
                "n": n,
                "flips": flips,
                "rate": flips / n if n else 0.0,
            }
        )
    return rows


def _depth(cfg: dict[str, Any], data: Dataset) -> list[dict[str, Any]]:
    rcfg = cfg["robustness"]
    archs = rcfg["depth_mlp"] + rcfg["depth_cos"] + rcfg["depth_kan"]
    lrs, _ = select_learning_rates(archs, cfg, data)
    runs = train_seeds(archs, lrs, cfg, data)
    rows = []
    for arch in archs:
        stats: list[dict[str, Any]] = []
        for run in runs[arch]:
            ref = TorchBackend(run.model).predict_logits(data.x_test)
            q = HybridExecutor(run.model).predict_logits(data.x_test)
            b = TorchBackend(run.model, torch.bfloat16).predict_logits(data.x_test)
            stats.append(
                {
                    "f1": _f1(data.y_test, ref),
                    "toolkit": output_agreement(ref, q),
                    "bf16": output_agreement(ref, b),
                }
            )
        spec = ArchSpec.parse(arch)
        rows.append(
            {
                "arch": arch,
                "family": spec.family,
                "depth": len(spec.hidden) + 1,
                "f1_fp32": mean_std([s["f1"] for s in stats]),
                "rel_rms_toolkit": mean_std([s["toolkit"]["rel_rms_diff"] for s in stats]),
                "rel_rms_bf16": mean_std([s["bf16"]["rel_rms_diff"] for s in stats]),
                "flips_toolkit": mean_std([s["toolkit"]["decision_flips"] for s in stats]),
            }
        )
    return rows


def run(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    data = dataset_from(cfg)
    arch = json.loads((out / "architecture.json").read_text(encoding="utf-8"))
    names = _study_models(arch)
    runs = train_seeds(names, arch["training"]["selected_learning_rates"], cfg, data)
    models = {}
    for name in names:
        reports = [_seed_report(r, data, cfg["robustness"], i) for i, r in enumerate(runs[name])]
        models[name] = _pool(reports)
    result = {
        "hypotheses": __doc__.split("\n\n")[1],
        "dataset": data.summary(),
        "models": models,
        "depth": _depth(cfg, data),
    }
    write_json(out / "robustness.json", result)
    return result
