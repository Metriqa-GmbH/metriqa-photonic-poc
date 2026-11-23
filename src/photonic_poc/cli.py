"""Command-line entry point: ``photonic-poc <command>``. Run ``photonic-poc -h`` for help."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from photonic_poc.benchmark import environment_info, noise_sweep, run_benchmark
from photonic_poc.data import GeneratorConfig, generate_splits, load_pairs, save_pairs
from photonic_poc.features import FEATURE_NAMES, featurize
from photonic_poc.inference import BACKEND_NAMES
from photonic_poc.models import MODEL_NAMES
from photonic_poc.reporting import render_results
from photonic_poc.training import TrainConfig, load_checkpoint, save_checkpoint, train_model

SPLITS = ("train", "val", "test")
# Noise-aware training is pointless for logreg: its only noise site is the output logit.
NOISE_TRAINED_MODELS = ("mlp", "fourier")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cmd_generate(args: argparse.Namespace) -> None:
    config = GeneratorConfig(n_entities=args.entities, seed=args.seed)
    splits = generate_splits(config)
    args.data.mkdir(parents=True, exist_ok=True)
    for split, pairs in splits.items():
        save_pairs(args.data / f"{split}.jsonl", pairs)
        print(f"{split}: {len(pairs)} pairs, {sum(p.label for p in pairs)} matches")
    _write_json(args.data / "generator.json", asdict(config))


def _load_split(data: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pairs = load_pairs(data / f"{split}.jsonl")
    x, y = featurize(pairs)
    return x, y, [p.kind for p in pairs]


def _artifact_label(model: str, noise_std: float) -> str:
    return model if noise_std == 0 else f"{model}-noise{noise_std:g}"


def cmd_train(args: argparse.Namespace) -> None:
    x_train, y_train, _ = _load_split(args.data, "train")
    x_val, y_val, _ = _load_split(args.data, "val")
    config = TrainConfig(epochs=args.epochs, seed=args.seed, noise_std=args.noise_std)
    summary = {}
    for name in args.models:
        result = train_model(name, x_train, y_train, x_val, y_val, config)
        label = _artifact_label(name, args.noise_std)
        meta = {"n_features": x_train.shape[1], "feature_names": list(FEATURE_NAMES)}
        save_checkpoint(args.artifacts / f"{label}.pt", name, result, meta | asdict(config))
        summary[label] = {"best_epoch": result.best_epoch, "val": result.val_metrics}
        print(f"{label}: best epoch {result.best_epoch}, val F1 {result.val_metrics['f1']:.4f}")
    existing = args.results / "training.json"
    previous = json.loads(existing.read_text()) if existing.exists() else {}
    _write_json(existing, previous | summary)


def cmd_evaluate(args: argparse.Namespace) -> None:
    x_test, y_test, kinds = _load_split(args.data, "test")
    loaded = [(path.stem, *load_checkpoint(path)[:2]) for path in args.artifacts.glob("*.pt")]
    loaded.sort(key=lambda item: (MODEL_NAMES.index(item[1]), item[0]))
    models = {label: model for label, _, model in loaded}
    if not models:
        raise SystemExit(f"no checkpoints in {args.artifacts}; run `photonic-poc train` first")
    report = run_benchmark(models, x_test, y_test, args.backends, kinds, args.latency_repeats)
    _write_json(args.results / "benchmark.json", report)
    base = {k: v for k, v in models.items() if "-noise" not in k}
    _write_json(
        args.results / "noise_sweep.json",
        noise_sweep(models if args.all_noise else base, x_test, y_test),
    )
    for row in report["rows"]:
        f1, flips = row["metrics"]["f1"], row["agreement_vs_fp32"]["decision_flips"]
        print(f"{row['model']:>18} {row['backend']:>11}  F1 {f1:.4f}  flips {flips}")


def cmd_op_numerics(args: argparse.Namespace) -> None:
    from photonic_poc.op_numerics import run_op_study

    _write_json(args.results / "op_numerics.json", run_op_study())
    print(f"wrote {args.results / 'op_numerics.json'}")


def cmd_sdk_info(_: argparse.Namespace) -> None:
    from photonic_poc.qant_backend import import_toolkit

    qant = import_toolkit()
    npus = qant.generic.get_available_npus()
    print(json.dumps(environment_info(), indent=2))
    for device_id, serial in npus:
        caps = [str(c).split(".")[-1] for c in qant.info.get_capabilities(device_id)]
        print(f"NPU {device_id} ({serial}): {', '.join(caps)}")


def cmd_seed_study(args: argparse.Namespace) -> None:
    from photonic_poc.seed_study import run_seed_study

    splits = {s: _load_split(args.data, s)[:2] for s in SPLITS}
    variants = [(m, 0.0) for m in MODEL_NAMES] + [
        (m, args.train_noise) for m in NOISE_TRAINED_MODELS
    ]
    rows = run_seed_study(
        variants, splits["train"], splits["val"], splits["test"], range(args.seeds), args.epochs,
        use_qant=not args.no_qant,
    )  # fmt: skip
    _write_json(args.results / "seed_study.json", rows)
    print(f"wrote {args.results / 'seed_study.json'}")


def cmd_research(args: argparse.Namespace) -> None:
    """Run the research experiments (A, B, C) from a TOML config, then render the figures."""
    from photonic_poc.experiments import amd_npu, architecture, hybrid_cost, robustness
    from photonic_poc.experiments.config import load_config
    from photonic_poc.figures import render_figures

    cfg = load_config(args.config)
    steps = {
        "architecture": architecture.run,
        "robustness": robustness.run,
        "hybrid": hybrid_cost.run,
        "amd-npu": amd_npu.run,
    }
    for name in args.only or [s for s in steps if s != "amd-npu"]:  # amd-npu: opt-in, Windows only
        print(f"== experiment {name}")
        steps[name](cfg, args.out)
    for path in render_figures(args.out, args.out / "figures"):
        print(f"wrote {path}")


def cmd_figures(args: argparse.Namespace) -> None:
    from photonic_poc.figures import render_figures

    for path in render_figures(args.out, args.out / "figures"):
        print(f"wrote {path}")


def cmd_report(args: argparse.Namespace) -> None:
    (args.results / "RESULTS.md").write_text(render_results(args.results), encoding="utf-8")
    print(f"wrote {args.results / 'RESULTS.md'}")


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Full pipeline with the published settings; --quick is a fast smoke run for CI."""
    entities, epochs = (600, 5) if args.quick else (args.entities, args.epochs)
    common = {"data": args.data, "artifacts": args.artifacts, "results": args.results}
    # Start from a clean slate so stale checkpoints of earlier runs are not benchmarked.
    for stale in [*args.artifacts.glob("*.pt"), args.results / "training.json"]:
        stale.unlink(missing_ok=True)
    cmd_generate(argparse.Namespace(**common, entities=entities, seed=args.seed))
    cmd_train(
        argparse.Namespace(**common, models=list(MODEL_NAMES), epochs=epochs, seed=0, noise_std=0.0)
    )
    if args.train_noise:
        noisy = argparse.Namespace(
            **common, models=list(NOISE_TRAINED_MODELS), epochs=epochs, seed=0
        )
        cmd_train(argparse.Namespace(**vars(noisy), noise_std=args.train_noise))
    backends = [b for b in BACKEND_NAMES if b != "qant" or not args.no_qant]
    cmd_evaluate(
        argparse.Namespace(
            **common, backends=backends, latency_repeats=2 if args.quick else 5, all_noise=True
        )
    )
    if not args.no_qant:
        cmd_op_numerics(argparse.Namespace(**common))
    seeds = 2 if args.quick else args.seeds
    cmd_seed_study(
        argparse.Namespace(
            **common,
            seeds=seeds,
            epochs=epochs,
            train_noise=args.train_noise or 0.05,
            no_qant=args.no_qant,
        )
    )
    cmd_report(argparse.Namespace(**common))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photonic-poc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Any, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        p.add_argument("--results", type=Path, default=Path("results"))
        return p

    p = add("generate", cmd_generate, "write the synthetic train/val/test pairs")
    p.add_argument("--entities", type=int, default=GeneratorConfig.n_entities)
    p.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    p = add("train", cmd_train, "train models on the generated data")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--noise-std", type=float, default=0.0, help="noise-aware training")
    p = add("evaluate", cmd_evaluate, "benchmark all checkpoints on the test split")
    p.add_argument("--backends", nargs="+", choices=BACKEND_NAMES, default=list(BACKEND_NAMES))
    p.add_argument("--latency-repeats", type=int, default=5)
    p.add_argument(
        "--all-noise", action="store_true", help="noise sweep for noise-trained models too"
    )
    add("op-numerics", cmd_op_numerics, "operation-level numerics of the toolkit")
    add("sdk-info", cmd_sdk_info, "show toolkit version, driver and NPUs")
    p = add("seed-study", cmd_seed_study, "retrain over several seeds; F1 spread per backend")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--train-noise", type=float, default=0.05)
    p.add_argument("--no-qant", action="store_true")
    add("report", cmd_report, "render RESULTS.md from the JSON results")
    p = add("research", cmd_research, "run the research experiments (see configs/research.toml)")
    p.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    p.add_argument("--only", nargs="+", choices=["architecture", "robustness", "hybrid", "amd-npu"])
    p.add_argument("--out", type=Path, default=Path("results/research"))
    p = add("figures", cmd_figures, "render the research figures from results/research")
    p.add_argument("--out", type=Path, default=Path("results/research"))
    from photonic_poc.private_eval import register_cli

    register_cli(sub)
    p = add("reproduce", cmd_reproduce, "run the whole pipeline")
    p.add_argument("--entities", type=int, default=GeneratorConfig.n_entities)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    p.add_argument("--train-noise", type=float, default=0.05, help="0 disables noise-aware runs")
    p.add_argument("--seeds", type=int, default=5, help="training seeds for the seed study")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-qant", action="store_true", help="skip the toolkit (not installed)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
