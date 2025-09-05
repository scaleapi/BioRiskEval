#!/usr/bin/env python3
"""Evaluate best saved DMS probes on the test split and save results.

This script loads the best probe per results CSV based on selection strategy
(mean-absolute val Spearman and/or train RMSE), evaluates on the corresponding
test split, and writes a results CSV.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from train_dms_probe import read_probe_dataset, evaluate_probe_continuous, select_best_row


def _infer_suffix(csv_path: Path) -> str:
    stem = csv_path.stem
    m = re.match(r"probe_results_(.+)", stem)
    return m.group(1) if m else "original"


def _pick_model_subdir(datasets_root: Path, suffix: str) -> Optional[Path]:
    if not datasets_root.exists():
        return None
    subdirs = [p for p in datasets_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    if suffix != "original":
        m = re.match(r"(\d+)steps", suffix)
        if m:
            steps = m.group(1)
            preferred = [p for p in subdirs if f"_1m_{steps}_" in p.name]
            if preferred:
                return sorted(preferred)[0]
            fallback = [p for p in subdirs if f"_{steps}_" in p.name or p.name.endswith(steps)]
            if fallback:
                return sorted(fallback)[0]
    nemo_like = [p for p in subdirs if "nemo2" in p.name]
    if nemo_like:
        return sorted(nemo_like)[0]
    no_steps = [p for p in subdirs if not re.search(r"_1m_\d+_", p.name)]
    if no_steps:
        return sorted(no_steps)[0]
    return sorted(subdirs)[0]


def _parse_layer_from_token(token: str) -> Optional[int]:
    m = re.search(r"layer[_=]?(\d+)", token)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _rebuild_probe_from_artifacts(weights_path: Path) -> torch.nn.Module:
    """Load a saved probe .pt and its metadata .json; rebuild the model and load weights."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    metadata_path = weights_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found next to weights: {metadata_path}")
    state = torch.load(str(weights_path), map_location="cpu")
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    in_features = int(state.get("in_features"))
    out_features = int(state.get("out_features"))
    arch = state.get("arch", "linear")
    if arch == "linear" or not metadata.get("non_linear", False):
        model = torch.nn.Linear(in_features, out_features)
    else:
        # Build MLP from metadata
        hidden_dims_raw = metadata.get("hidden_dims", "512") or "512"
        if isinstance(hidden_dims_raw, str):
            hidden_dims = [int(h) for h in hidden_dims_raw.split(",") if h.strip()]
        elif isinstance(hidden_dims_raw, (list, tuple)):
            hidden_dims = [int(h) for h in hidden_dims_raw]
        else:
            hidden_dims = [512]
        activation_name = str(metadata.get("activation", "relu")).lower()
        if activation_name == "relu":
            act_layer = torch.nn.ReLU
        elif activation_name == "gelu":
            act_layer = torch.nn.GELU
        elif activation_name == "tanh":
            act_layer = torch.nn.Tanh
        else:
            raise ValueError(f"Unsupported activation in metadata: {activation_name}")
        dropout_p = float(metadata.get("dropout", 0.0) or 0.0)
        layers: List[torch.nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(act_layer())
            if dropout_p > 0:
                layers.append(torch.nn.Dropout(p=dropout_p))
            prev = h
        layers.append(torch.nn.Linear(prev, out_features))
        model = torch.nn.Sequential(*layers)
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def _find_best_probe_weights(
    model_dir: Path,
    layer: Optional[int],
    selection_strategy: str,
) -> Optional[Path]:
    if not model_dir.exists():
        return None
    layer_token = str(layer) if layer is not None else "x"
    pattern = f"dms_probe_*_layer={layer_token}_*.pt"
    matches = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _evaluate_and_write(
    results_dir: Path,
    datasets_root: Path,
    probes_root: Path,
    selection: str,
    device: str,
    out_base: Path,
) -> None:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    csv_files = sorted(results_dir.glob("probe_results_*.csv"))
    if not csv_files:
        print(f"No probe_results_*.csv files found in {results_dir}")
        return
    out_dir = out_base / selection
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        suffix = _infer_suffix(csv_path)
        model_subdir = _pick_model_subdir(datasets_root, suffix)
        if model_subdir is None:
            print(f"Skipping {csv_path.name}: could not infer model subdir under {datasets_root}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path.name}: failed to read CSV ({e})")
            continue
        try:
            best = select_best_row(df, task_type="continuous", probe_pick=("train_rmse" if selection == "train_rmse" else "val_spearman"))
        except Exception as e:
            print(f"Skipping {csv_path.name}: selection failed ({e})")
            continue
        train_token = str(best.get("train_dataset_path", "")).strip()
        layer_inferred = _parse_layer_from_token(train_token)
        if layer_inferred is None:
            print(f"Skipping {csv_path.name}: could not parse layer from train token '{train_token}'")
            continue
        test_path = model_subdir / f"dms_probe_dataset_layer_{layer_inferred}_test.h5"
        if not test_path.exists():
            print(f"Skipping {csv_path.name}: test dataset not found: {test_path}")
            continue
        subfolder = "closed_form_train_rmse" if selection == "train_rmse" else "closed_form_val_spearman"
        model_dir = probes_root / subfolder / model_subdir.name
        weights_path = _find_best_probe_weights(model_dir, layer_inferred, selection)
        if weights_path is None:
            print(f"Skipping {csv_path.name}: no saved weights found in {model_dir} for layer={layer_inferred}")
            continue
        print(f"Evaluating {weights_path.name} on {test_path.name} [{selection}] ...")
        model = _rebuild_probe_from_artifacts(weights_path).to(device)
        _, test_repr, test_labels, test_datasets = read_probe_dataset(str(test_path))
        rmse, mae, r2, pearson, avg_spearman, per_dataset = evaluate_probe_continuous(model, test_repr, test_labels, test_datasets, None)
        # Write per-CSV results file grouped by suffix
        out_csv = out_dir / f"probe_results_{suffix}.csv"
        file_exists = out_csv.exists()
        with out_csv.open("a", newline="") as f:
            import csv as _csv
            writer = _csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "train_dataset_path",
                    "test_dataset_path",
                    "task_type",
                    "loss_function",
                    "final_train_loss",
                    "train_rmse",
                    "test_rmse",
                    "test_mae",
                    "test_r2",
                    "test_pearson",
                    "avg_spearman",
                    "per_dataset_spearman",
                    "weights_path",
                ])
            per_dataset_json = "{}"
            if per_dataset is not None:
                valid_map = {}
                for k, v in per_dataset.items():
                    key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                    if v is None or np.isnan(v):
                        continue
                    valid_map[key_str] = float(v)
                per_dataset_json = json.dumps(valid_map, separators=(",", ":"))
            # Safely convert metrics that might be None or non-numeric strings
            def _safe_float(val: object) -> Optional[float]:
                try:
                    return float(val)  # type: ignore[arg-type]
                except Exception:
                    return None

            def _to_float_or_empty(val: object) -> str:
                v = _safe_float(val)
                return f"{v:.6f}" if v is not None else ""

            writer.writerow([
                str(best.get("train_dataset_path", "")),
                test_path.stem,
                "continuous",
                str(best.get("loss_function", "mse")),
                _to_float_or_empty(best.get('final_train_loss', 0.0)),
                _to_float_or_empty(best.get('train_rmse', None)),
                f"{rmse:.6f}",
                f"{mae:.6f}",
                f"{r2:.6f}",
                f"{pearson:.6f}",
                f"{avg_spearman:.6f}",
                per_dataset_json,
                str(weights_path),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best saved DMS probes on test split")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing probe_results_*.csv files (from sweep)")
    parser.add_argument("--datasets_root", type=str, required=True, help="Root directory of probe_datasets (contains per-model subfolders)")
    parser.add_argument("--probes_root", type=str, default="ft-attack/probe/saved_probes_stratified/k=500_seed=42", help="Root directory where best probes were saved")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device for evaluation")
    parser.add_argument("--eval_best_by_train_rmse", action="store_true", help="Evaluate the best-by-train-RMSE saved probe")
    parser.add_argument("--eval_best_by_val_spearman", action="store_true", help="Evaluate the best-by-val-Spearman saved probe")
    parser.add_argument("--out_dir", type=str, default=None, help="Base directory to write test-split results (derived from probes_root if omitted)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets_root = Path(args.datasets_root)
    probes_root = Path(args.probes_root)
    # Derive default out_dir from probes_root naming convention if not provided.
    # Example: ft-attack/probe/saved_probes_stratified/k=500_seed=42 -> probe_results_stratified/k=500_seed=42/test
    if args.out_dir is None:
        pr = Path(args.probes_root)
        parts = pr.parts
        idx = None
        for i, seg in enumerate(parts):
            if seg.startswith("saved_probes"):
                idx = i
                break
        if idx is not None:
            saved_name = parts[idx]
            replaced_name = saved_name.replace("saved_probes", "probe_results")
            tail = Path(*parts[idx + 1:]) if (idx + 1) < len(parts) else Path("")
            out_base = Path(replaced_name) / tail / "test"
        else:
            out_base = Path("probe_results/test")
    else:
        out_base = Path(args.out_dir)

    if not args.eval_best_by_train_rmse and not args.eval_best_by_val_spearman:
        # Default to val_spearman if no flag provided
        args.eval_best_by_val_spearman = True

    if args.eval_best_by_train_rmse:
        _evaluate_and_write(results_dir, datasets_root, probes_root, selection="train_rmse", device=args.device, out_base=out_base)
    if args.eval_best_by_val_spearman:
        _evaluate_and_write(results_dir, datasets_root, probes_root, selection="val_spearman", device=args.device, out_base=out_base)


if __name__ == "__main__":
    main()


