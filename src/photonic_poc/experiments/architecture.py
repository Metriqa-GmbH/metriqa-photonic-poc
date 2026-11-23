"""Experiment A: do photonic-compatible architectures reach MLP quality with fewer parameters/ops?

H_A: at matched test F1 (within seed and sampling noise), Fourier-KAN or cosine networks need
fewer parameters or operations than ReLU MLPs. Rejected if MLPs reach every quality level at
equal or lower cost. Architecture effects are measured in float32; the toolkit simulator is
evaluated separately so backend effects cannot masquerade as architecture effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from photonic_poc.datasets import Dataset
from photonic_poc.experiments.config import (
    dataset_from,
    mean_std,
    select_learning_rates,
    train_seeds,
    write_json,
)
from photonic_poc.inference import create_backend
from photonic_poc.metrics import classification_metrics, output_agreement
from photonic_poc.models import FAMILIES, ArchSpec
from photonic_poc.op_profile import profile_model
from photonic_poc.statistics import paired_bootstrap_f1, pareto_front, spearman
from photonic_poc.sweep import Trained

PROFILE_KEYS = (
    "macs",
    "nonlin_evals",
    "total_ops",
    "npu_ops",
    "offload_fraction",
    "npu_calls",
    "transfer_elements",
)


def _evaluate(
    arch: str, runs: list[Trained], data: Dataset
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    preds: dict[str, list[np.ndarray]] = {"fp32": [], "qant": []}
    rows: dict[str, list[dict[str, float]]] = {"fp32": [], "qant": [], "agree": []}
    val_f1 = []
    for run in runs:
        fp32 = create_backend("torch-fp32", run.model).predict_logits(data.x_test)
        qant = create_backend("qant", run.model).predict_logits(data.x_test)
        val = create_backend("torch-fp32", run.model).predict_logits(data.x_val)
        val_f1.append(classification_metrics(data.y_val, val)["f1"])
        rows["fp32"].append(classification_metrics(data.y_test, fp32))
        rows["qant"].append(classification_metrics(data.y_test, qant))
        rows["agree"].append(output_agreement(fp32, qant))
        preds["fp32"].append(fp32 > 0)
        preds["qant"].append(qant > 0)
    spec = ArchSpec.parse(arch)
    profile = profile_model(runs[0].model)
    record: dict[str, Any] = {
        "arch": arch,
        "family": spec.family,
        "learning_rate": runs[0].job.lr,
        "parameters": profile["parameters"],
        **{k: profile[k] for k in PROFILE_KEYS},
        "val_f1": mean_std(val_f1),
        "train_seconds": mean_std([r.train_seconds for r in runs]),
        "best_epoch": mean_std([r.best_epoch for r in runs]),
    }
    for backend in ("fp32", "qant"):
        for metric in ("f1", "precision", "recall", "roc_auc"):
            record[f"{backend}_{metric}"] = mean_std([m[metric] for m in rows[backend]])
    deltas = [q["f1"] - f["f1"] for q, f in zip(rows["qant"], rows["fp32"], strict=True)]
    record["delta_f1"] = mean_std(deltas)
    record["max_abs_delta_f1"] = float(np.max(np.abs(deltas)))
    record["flips"] = mean_std([a["decision_flips"] for a in rows["agree"]])
    record["rel_rms_logit_dev"] = mean_std([a["rel_rms_diff"] for a in rows["agree"]])
    return record, {k: np.stack(v) for k, v in preds.items()}


def _gbdt_reference(data: Dataset, seeds: list[int]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    f1s = []
    for seed in seeds:
        clf = HistGradientBoostingClassifier(random_state=seed).fit(data.x_train, data.y_train)
        logits = clf.decision_function(data.x_test)
        f1s.append(classification_metrics(data.y_test, logits)["f1"])
    return {"model": "HistGradientBoostingClassifier (scikit-learn defaults)", "f1": mean_std(f1s)}


def _champions(records: list[dict[str, Any]], tolerance: float) -> dict[str, dict[str, str]]:
    """Per family: best mean validation F1, and smallest model within ``tolerance`` of the best."""
    best_val = max(r["val_f1"]["mean"] for r in records)
    out: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        members = [r for r in records if r["family"] == family]
        if not members:
            continue
        champion = max(members, key=lambda r: r["val_f1"]["mean"])
        compact = [r for r in members if r["val_f1"]["mean"] >= best_val - tolerance]
        out[family] = {"best": champion["arch"]}
        if compact:
            out[family]["compact"] = min(compact, key=lambda r: r["parameters"])["arch"]
    return out


def _iso_quality(records: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    best = max(r[f"{backend}_f1"]["mean"] for r in records)
    rows = []
    for margin in (0.002, 0.005, 0.01, 0.02):
        target = best - margin
        row: dict[str, Any] = {"margin": margin, "target_f1": target}
        for family in FAMILIES:
            ok = [
                r for r in records if r["family"] == family and r[f"{backend}_f1"]["mean"] >= target
            ]
            row[family] = (
                {k: min(r[k] for r in ok) for k in ("parameters", "total_ops", "npu_ops")}
                if ok
                else None
            )
        rows.append(row)
    return rows


def run(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    data = dataset_from(cfg)
    acfg = cfg["architecture"]
    archs = [a for family in FAMILIES for a in acfg.get(family, [])]
    lrs, _ = select_learning_rates(archs, cfg, data)
    runs = train_seeds(archs, lrs, cfg, data)
    records, preds = [], {}
    for arch in archs:
        record, arch_preds = _evaluate(arch, runs[arch], data)
        records.append(record)
        preds[arch] = arch_preds
    champions = _champions(records, acfg["compact_tolerance"])
    comparisons = []
    pairs = [("kan", "mlp"), ("cos", "mlp"), ("mlp", "logreg")]
    for kind in ("best", "compact"):
        for fa, fb in pairs:
            a, b = champions.get(fa, {}).get(kind), champions.get(fb, {}).get(kind)
            if a is None or b is None:
                continue
            for backend in ("fp32", "qant"):
                stats = paired_bootstrap_f1(
                    data.y_test, preds[a][backend], preds[b][backend], acfg["bootstrap"]
                )
                comparisons.append({"selection": kind, "a": a, "b": b, "backend": backend, **stats})
    fronts = {}
    for backend in ("fp32", "qant"):
        scores = [r[f"{backend}_f1"]["mean"] for r in records]
        for cost in ("parameters", "total_ops", "npu_ops"):
            idx = pareto_front([r[cost] for r in records], scores)
            fronts[f"{backend}:{cost}"] = [records[i]["arch"] for i in idx]
    fp32_scores = [r["fp32_f1"]["mean"] for r in records]
    qant_scores = [r["qant_f1"]["mean"] for r in records]
    result = {
        "hypothesis": __doc__.split("\n\n")[1].replace("\n", " "),
        "dataset": data.summary(),
        "training": cfg["training"] | {"selected_learning_rates": lrs},
        "trainings_run": len(archs)
        * (len(cfg["training"]["learning_rates"]) + len(cfg["training"]["seeds"]) - 1),
        "records": records,
        "champions": champions,
        "comparisons": comparisons,
        "pareto_fronts": fronts,
        "iso_quality": {b: _iso_quality(records, b) for b in ("fp32", "qant")},
        "backend_effect": {
            "spearman_fp32_vs_qant": spearman(fp32_scores, qant_scores),
            "max_abs_mean_delta_f1": float(np.max(np.abs(np.subtract(qant_scores, fp32_scores)))),
            "fronts_identical": {
                cost: fronts[f"fp32:{cost}"] == fronts[f"qant:{cost}"]
                for cost in ("parameters", "total_ops", "npu_ops")
            },
        },
        "reference": _gbdt_reference(data, cfg["training"]["seeds"]),
    }
    write_json(out / "architecture.json", result)
    return result
