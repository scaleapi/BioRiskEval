#!/usr/bin/env python3
"""
Test a saved DMS probe.

Two modes:
1) Direct: provide --probe_path and --test_dataset_path.
2) From CSV: provide --results_csv to select the best row and reconstruct the probe path
   by searching under --probes_root using the naming convention.

Notes:
- For binary tasks we report accuracy and AUC.
- For continuous tasks we report RMSE, MAE, R2, Pearson, and avg Spearman.
- For nonlinear probes, we rebuild the MLP from metadata (hidden_dims/activation/dropout).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import csv

import torch
import pandas as pd

# Reuse dataset/eval helpers from training script
from train_probe_dms import (
    read_probe_dataset,
    evaluate_probe,
    evaluate_probe_continuous,
)
try:
    from probe.results_selection import select_best_row  # when run as module
except Exception:
    from results_selection import select_best_row  # when run as script


DEFAULT_PROBES_ROOT = Path("ft-attack/probe/saved_probes")


def _parse_layer_from_name(name: str) -> Optional[int]:
    match = re.search(r"layer[\s_\-=:]?(\d+)", name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _build_search_pattern(task_type: str,
                          is_nonlinear: bool,
                          mode_token: str,
                          layer: Optional[int],
                          loss_name: str,
                          learning_rate: float,
                          batch_size: int) -> str:
    lr_token = f"{float(learning_rate):.10g}"
    layer_token = str(layer) if layer is not None else "x"
    nonlinear_token = "_nonlinear" if is_nonlinear else ""
    # Seed is wildcard; we pick newest match
    return (
        f"dms_probe_{task_type}{nonlinear_token}_{mode_token}_layer={layer_token}"
        f"_loss={loss_name}_lr={lr_token}_batch_size={batch_size}_seed=*.pt"
    )


def _rebuild_probe_from_artifacts(weights_path: Path) -> torch.nn.Module:
    """Load a saved probe .pt and its metadata .json; rebuild the model and load weights."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    # Infer metadata path
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


def _select_best_row(df: pd.DataFrame, task_type: str, probe_pick: Optional[str] = None) -> pd.Series:
    return select_best_row(df, task_type, probe_pick)


def _to_int_or_none(x: object) -> Optional[int]:
    if x is None:
        return None
    try:
        s = str(x)
        if s == "None" or s.strip() == "":
            return None
        # Many CSVs store ints as strings or floats; coerce safely
        return int(float(s))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a saved DMS probe (direct or from CSV)")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--probe_path", type=str, help="Path to saved probe .pt/.pth file")
    mode.add_argument("--results_csv", type=str, help="CSV of probe results to pick best row from")
    mode.add_argument("--results_dir", type=str, default=None, help="Directory containing probe_results_*.csv files to iterate over")

    parser.add_argument("--task_type", type=str, choices=["binary", "continuous"], help="Task type (required for direct mode; optional for CSV if present in file)")
    parser.add_argument("--test_dataset_path", type=str, help="Path to test .h5 dataset (required for direct mode)")
    parser.add_argument("--datasets_root", type=str, default="data/eval_dataset/fitness/probe_datasets", help="Root directory containing .h5 datasets when using CSV mode")
    parser.add_argument("--probes_root", type=str, default="probe/saved_probes", help="Root directory to search for saved probes when using CSV mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="Device to run evaluation")
    parser.add_argument("--probe_pick", type=str, choices=["train_rmse", "val_spearman"], default="val_spearman", help="How to select best probe row from CSV for continuous tasks")

    args = parser.parse_args()

    if args.probe_path:
        # Direct mode
        if not args.test_dataset_path:
            raise ValueError("--test_dataset_path is required when using --probe_path")
        if not args.task_type:
            raise ValueError("--task_type is required when using --probe_path")

        weights_path = Path(args.probe_path)
        model = _rebuild_probe_from_artifacts(weights_path).to(args.device)

        # Load dataset
        _, test_repr, test_labels, test_datasets = read_probe_dataset(args.test_dataset_path)
        # Evaluate
        if args.task_type == "binary":
            acc, auc = evaluate_probe(model, test_repr, test_labels, None)
            print(f"Test acc: {acc:.4f}\nTest AUC: {auc:.4f}")
        else:
            rmse, mae, r2, pearson, avg_spearman, per_ds = evaluate_probe_continuous(model, test_repr, test_labels, test_datasets, None)
            print(f"Test RMSE: {rmse:.6f}\nTest MAE: {mae:.6f}\nTest R2: {r2:.6f}\nTest Pearson: {pearson:.6f}\nAvg Spearman: {avg_spearman:.6f}")
        return

    # CSV mode(s): single file or directory of files
    def _process_results_csv(results_csv_path: Path) -> None:
        if not results_csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")
        print(f"[debug] processing CSV: {results_csv_path}")
        df_local = pd.read_csv(results_csv_path)

        local_task_type = args.task_type or (df_local["task_type"].iloc[0] if "task_type" in df_local.columns else None)
        if local_task_type not in {"binary", "continuous"}:
            raise ValueError("task_type must be provided via --task_type or exist in CSV column 'task_type'")

        best_local = _select_best_row(df_local, local_task_type, args.probe_pick)

        # Extract fields
        train_name_local = str(best_local["train_dataset_path"]) if "train_dataset_path" in best_local else ""
        test_name_local = str(best_local["test_dataset_path"]) if "test_dataset_path" in best_local else ""
        lr_local = float(best_local["learning_rate"]) if "learning_rate" in best_local else None
        bs_local = int(best_local["batch_size"]) if "batch_size" in best_local else None
        num_steps_local = _to_int_or_none(best_local.get("num_steps", None))
        num_epochs_local = _to_int_or_none(best_local.get("num_epochs", None))
        if num_epochs_local is not None:
            mode_token_local = f"epochs={int(num_epochs_local)}"
        elif num_steps_local is not None:
            mode_token_local = f"steps={int(num_steps_local)}"
        else:
            mode_token_local = "closed_form" if local_task_type == "continuous" else "steps=UNKNOWN"

        layer_local = _parse_layer_from_name(train_name_local) or _parse_layer_from_name(test_name_local)
        print(f"[debug] selected train token={train_name_local} | parsed_layer={layer_local}")

        # Detect closed_form CSVs explicitly (these include a train_rmse column)
        is_closed_form_local = ("train_rmse" in df_local.columns)

        if local_task_type == "binary":
            loss_name_local = "bce"
            is_nonlinear_local = False
            subfolder_local = "linear"
        else:
            is_nonlinear_local = False
            if "non_linear" in df_local.columns and pd.notna(best_local.get("non_linear")):
                try:
                    is_nonlinear_local = str(best_local.get("non_linear")).lower() == "true"
                except Exception:
                    is_nonlinear_local = False
            loss_name_local = str(best_local.get("loss_function", "mse"))
            if is_closed_form_local:
                # Force closed_form routing regardless of num_steps/num_epochs contents in CSV
                mode_token_local = "closed_form"
                subfolder_local = "closed_form"
                loss_name_local = "mse"
            else:
                subfolder_local = "nonlinear" if is_nonlinear_local else "continuous"

        # Force the search layer to the parsed layer from the selected row (do not rely on filename layer)
        forced_layer = layer_local

        if lr_local is None or bs_local is None:
            raise ValueError("CSV missing learning_rate or batch_size")

        # Build search pattern (we will choose search roots after inferring model subdir)
        # Build search pattern using the forced_layer to match the selected row's layer
        pattern_local = _build_search_pattern(
            local_task_type,
            is_nonlinear_local,
            mode_token_local,
            forced_layer,
            loss_name_local,
            lr_local,
            bs_local,
        )

        if not args.datasets_root:
            raise ValueError("--datasets_root is required in CSV mode to locate .h5 datasets")

        # Suffix and model subdir
        stem_local = results_csv_path.stem
        mm = re.match(r"probe_results_(.+)", stem_local)
        suffix_local = mm.group(1) if mm else "original"

        datasets_root_local = Path(args.datasets_root)
        subdirs_local = [p for p in datasets_root_local.iterdir() if p.is_dir()]
        model_subdir_local = None
        if suffix_local != "original":
            smm = re.match(r"(\d+)steps", suffix_local)
            if smm:
                steps_local = smm.group(1)
                # Look for folders where the number after the third _ matches the steps
                for p in subdirs_local:
                    parts = p.name.split('_')
                    if len(parts) >= 4:  # Need at least 4 parts to have a third _
                        try:
                            if parts[3] == steps_local:  # Fourth part (index 3) should match steps
                                model_subdir_local = p
                                break
                        except (IndexError, ValueError):
                            continue
                
                # Fallback: original logic for compatibility
                if model_subdir_local is None:
                    preferred_local = [p for p in subdirs_local if f"_1m_{steps_local}_" in p.name]
                    if preferred_local:
                        model_subdir_local = sorted(preferred_local)[0]
                    else:
                        # Flexible fallback: any folder that mentions the steps token
                        fallback_local = [p for p in subdirs_local if f"_{steps_local}_" in p.name or p.name.endswith(steps_local)]
                        if fallback_local:
                            model_subdir_local = sorted(fallback_local)[0]
            # If still not found, last-resort: pick a single non-base folder so evaluation can proceed
            if model_subdir_local is None:
                non_base = [p for p in subdirs_local if re.search(r"_1m_\d+_", p.name)]
                if len(non_base) == 1:
                    model_subdir_local = non_base[0]
            if model_subdir_local is None:
                print(f"[warn] {results_csv_path.name}: could not strictly match '{suffix_local}', proceeding may fail to find data")
                # Do not return; allow probe fallback matching to try
        else:
            if model_subdir_local is None:
                nemo_like_local = [p for p in subdirs_local if "nemo2" in p.name]
                if nemo_like_local:
                    model_subdir_local = sorted(nemo_like_local)[0]
                else:
                    no_steps_local = [p for p in subdirs_local if not re.search(r"_1m_\d+_", p.name)]
                    model_subdir_local = sorted(no_steps_local)[0] if no_steps_local else (sorted(subdirs_local)[0] if subdirs_local else None)
            if model_subdir_local is None:
                raise FileNotFoundError(f"Could not infer model subdirectory under datasets_root for suffix '{suffix_local}'")

        # Resolve probe search roots: prefer per-model subfolder first
        probes_root_base = Path(args.probes_root) / subfolder_local
        search_roots: List[Path] = []
        if not probes_root_base.exists():
            raise FileNotFoundError(f"Probes root not found: {probes_root_base}")
        if model_subdir_local is None:
            print(f"Skipping {results_csv_path.name}: model subfolder could not be determined under datasets_root; not falling back to root probes")
            return
        candidate = probes_root_base / model_subdir_local.name
        if candidate.exists():
            search_roots.append(candidate)
        else:
            # Handle naming convention mismatch: try replacing = with - (or vice versa)
            alt_name = model_subdir_local.name.replace('=', '-')
            alt_candidate = probes_root_base / alt_name
            if alt_candidate.exists():
                search_roots.append(alt_candidate)
                print(f"[debug] found probes folder with modified name: {alt_candidate}")
            else:
                # Try the other direction too
                alt_name2 = model_subdir_local.name.replace('-', '=')
                alt_candidate2 = probes_root_base / alt_name2
                if alt_candidate2.exists():
                    search_roots.append(alt_candidate2)
                    print(f"[debug] found probes folder with modified name: {alt_candidate2}")
                else:
                    print(f"Skipping {results_csv_path.name}: probes folder not found at {candidate}, {alt_candidate}, or {alt_candidate2}")
                    return
        print(f"[debug] suffix={suffix_local} subfolder={subfolder_local} search_roots={search_roots}")

        matches_local = []
        for root in search_roots:
            found = sorted(root.glob(pattern_local), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                matches_local = found
                break
        # # Fallback: relax pattern to layer-only for closed_form probes
        # if not matches_local:
        #     relaxed_pattern = f"dms_probe_*_closed_form_layer={layer_local}_*.pt"
        #     for root in search_roots:
        #         found = sorted(root.glob(relaxed_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        #         if found:
        #             matches_local = found
        #             break
        if not matches_local:
            raise FileNotFoundError(f"No probe file matched pattern in {search_roots}: {pattern_local}")
        weights_path_local = matches_local[0]
        print(f"Selected probe: {weights_path_local}")

        model_local = _rebuild_probe_from_artifacts(weights_path_local).to(args.device)

        out_dir_local = Path("probe_results/test_split")
        out_dir_local.mkdir(parents=True, exist_ok=True)
        out_csv_local = out_dir_local / f"probe_results_{suffix_local}.csv"
        out_exists_local = out_csv_local.exists()

        # Evaluate only the layer the probe was trained on
        layers_to_eval_local: List[int] = []
        if layer_local is not None:
            try:
                layers_to_eval_local = [int(layer_local)]
            except Exception:
                layers_to_eval_local = []
        if not layers_to_eval_local:
            m_layer = re.search(r"layer=(\d+)", weights_path_local.name)
            if m_layer:
                try:
                    layers_to_eval_local = [int(m_layer.group(1))]
                except Exception:
                    layers_to_eval_local = []
        results_rows_local = []
        for layer_idx in layers_to_eval_local:
            rest_path_local = model_subdir_local / f"dms_probe_dataset_layer_{layer_idx}_rest.h5"
            if not rest_path_local.exists():
                print(f"Skipping missing REST dataset for layer {layer_idx}: {rest_path_local}")
                continue
            _, rest_repr_local, rest_labels_local, rest_datasets_local = read_probe_dataset(str(rest_path_local))
            if local_task_type == "binary":
                acc_local, auc_local = evaluate_probe(model_local, rest_repr_local, rest_labels_local, None)
                print(f"REST acc (layer {layer_idx}): {acc_local:.4f} | AUC: {auc_local:.4f}")
                continue
            rmse_local, mae_local, r2_local, pearson_local, avg_spearman_local, per_dataset_spearman_local = evaluate_probe_continuous(
                model_local, rest_repr_local, rest_labels_local, rest_datasets_local, None
            )
            per_dataset_json_local = "{}"
            avg_abs_local = 0.0  # Default value if no per-dataset data
            if per_dataset_spearman_local is not None:
                valid_map_local = {}
                abs_vals_local = []
                for k, v in per_dataset_spearman_local.items():
                    key_str_local = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                    if v is None or np.isnan(v):
                        continue
                    valid_map_local[key_str_local] = float(v)
                    abs_vals_local.append(abs(float(v)))
                if abs_vals_local:
                    avg_abs_local = float(np.mean(abs_vals_local))
                per_dataset_json_local = json.dumps(valid_map_local, separators=(",", ":"))

            lr_val_local = float(best_local["learning_rate"]) if "learning_rate" in best_local else 0.0
            bs_val_local = int(best_local["batch_size"]) if "batch_size" in best_local else 0
            loss_fn_local = str(best_local.get("loss_function", "mse"))
            final_train_loss_local = float(best_local.get("final_train_loss", 0.0)) if "final_train_loss" in best_local else 0.0
            train_rmse_local = best_local.get("train_rmse", "")
            train_path_str_local = str(best_local.get("train_dataset_path", ""))
            test_path_str_local = rest_path_local.stem

            row_local = [
                train_path_str_local,
                test_path_str_local,
                lr_val_local,
                bs_val_local,
                num_steps_local,
                num_epochs_local,
                "continuous",
                loss_fn_local,
                f"{final_train_loss_local:.6f}",
                f"{float(train_rmse_local):.6f}" if isinstance(train_rmse_local, (int, float)) else (str(train_rmse_local) if train_rmse_local else ""),
                f"{rmse_local:.6f}",
                f"{mae_local:.6f}",
                f"{r2_local:.6f}",
                f"{pearson_local:.6f}",
                f"{avg_abs_local:.6f}",
                per_dataset_json_local,
                "no_wandb",
            ]
            results_rows_local.append(row_local)

        if results_rows_local:
            with out_csv_local.open("a", newline="") as f:
                writer = csv.writer(f)
                if not out_exists_local:
                    writer.writerow([
                        "train_dataset_path",
                        "test_dataset_path",
                        "learning_rate",
                        "batch_size",
                        "num_steps",
                        "num_epochs",
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
                        "wandb_run",
                    ])
                for row_local in results_rows_local:
                    writer.writerow(row_local)
            print(f"Saved {len(results_rows_local)} REST evaluation rows to {out_csv_local}")
        else:
            print("No REST datasets found for the probe layer. Nothing written.")

    # Prefer explicit single-CSV mode when provided
    if args.results_csv:
        _process_results_csv(Path(args.results_csv))
        return

    if args.results_dir:
        base_dir = Path(args.results_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {base_dir}")
        csv_files = sorted(base_dir.glob("probe_results_*.csv"))
        if not csv_files:
            print(f"No probe_results_*.csv files found in {base_dir}")
            return
        for csv_path in csv_files:
            _process_results_csv(csv_path)
        return

    raise ValueError("Provide either --results_csv or --results_dir")


if __name__ == "__main__":
    main()


