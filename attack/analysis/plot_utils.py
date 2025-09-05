"""
Plotting/data utilities for probe analysis.

Responsibilities:
- Discover zero-shot models and read per-dataset absolute Spearman values
- Read probe closed_form CSVs and select the best row per model within a layer
- Provide unified outputs ready for plotting (no plotting here)

Design goals:
- One source of truth for layer filtering (train_dataset_path contains dms_probe_dataset_layer_{index}_train)
- One source of truth for probe row selection (max avg_spearman within the layer)
- No duplicated logic across notebooks and scripts
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Defaults can be overridden by callers
ZERO_SHOT_BASE_DIR_DEFAULT = (
    "/workspaces/src/models/bionemo-framework/attack/analysis/dms_results/likelihood/virus_reproduction/h5_samples=624_seed=42_test/Virus"
)
PROBE_RESULTS_CLOSED_FORM_DIR_DEFAULT = (
    "/workspaces/src/models/bionemo-framework/attack/analysis/dms_results/probe_results/closed_form"
)


# Preferred order for selecting a test-split metric from probe CSVs.
# Each tuple is (column_name, direction), where direction is 'min' or 'max'.
TEST_METRIC_PRIORITY_DEFAULT: List[Tuple[str, str]] = [
    ("test_rmse", "min"),
    ("test/rmse", "min"),
    ("test_mse", "min"),
    ("test/mse", "min"),
    ("test_mae", "min"),
    ("test/mae", "min"),
    ("test_spearman", "max"),
    # Fallback to avg_spearman which, for closed_form, reflects test performance summary
    ("avg_spearman", "max"),
]


def extract_steps_from_model_dir(model_dir_name: str) -> Optional[int]:
    """Extract fine-tuning steps from a zero-shot model directory name.

    Looks for pattern like "... 1m <steps> ...". Returns None for base/original.
    """
    parts = model_dir_name.split("_")
    for i, part in enumerate(parts):
        if part == "1m" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def list_zero_shot_models(zero_shot_base_dir: str) -> List[str]:
    """List model subdirectories sorted with original first then by steps."""
    if not os.path.isdir(zero_shot_base_dir):
        return []
    subdirs = [d for d in os.listdir(zero_shot_base_dir)
               if os.path.isdir(os.path.join(zero_shot_base_dir, d))]

    def sort_key(name: str):
        steps = extract_steps_from_model_dir(name)
        return (0, name) if steps is None else (1, steps)

    return sorted(subdirs, key=sort_key)


def read_zero_shot_spearman(zero_shot_base_dir: str) -> Dict[str, List[float]]:
    """Read per-dataset absolute Spearman from *_fitness.csv for each model.

    Returns mapping: model_dir_name -> List[abs(spearman)].
    """
    data: Dict[str, List[float]] = {}
    for model in list_zero_shot_models(zero_shot_base_dir):
        csv_paths = list(Path(zero_shot_base_dir, model).glob("*_fitness.csv"))
        values: List[float] = []
        for fp in csv_paths:
            try:
                df = pd.read_csv(fp)
                if "spearman" in df.columns:
                    v = df.loc[0, "spearman"]
                    if pd.notna(v):
                        values.append(abs(float(v)))
            except Exception:
                continue
        data[model] = values
    return data


def model_to_probe_suffix(model_dir_name: str) -> str:
    """Map a zero-shot model directory name to the probe_results suffix."""
    if "40b" in model_dir_name:
        print("40b in model_dir_name")
        return "40b"
    else:
        steps = extract_steps_from_model_dir(model_dir_name)
        return "original" if steps is None else f"{steps}steps"


def filter_df_by_layer_train_path(df: pd.DataFrame, layer_index: int) -> pd.DataFrame:
    """Filter a probe CSV DataFrame to rows for the given layer based on train_dataset_path.

    Layer index is zero-indexed in file names: conceptual layer N -> index N-1.
    """
    if df is None or df.empty or "train_dataset_path" not in df.columns:
        return df.iloc[0:0]
    pattern = rf"dms_probe_dataset_layer_{layer_index}_train"
    mask = df["train_dataset_path"].astype(str).str.contains(pattern, na=False)
    return df.loc[mask]


def _filter_df_by_layer_general(df: pd.DataFrame, layer_index: int) -> pd.DataFrame:
    """Filter rows belonging to a specific layer using available hints.

    Preference order:
    - Use train_dataset_path pattern (zero-indexed layer in file names)
    - Else, use a numeric 'layer' column if present (assumed to be integer)
    """
    if df is None or df.empty:
        return df.iloc[0:0]
    if "train_dataset_path" in df.columns:
        return filter_df_by_layer_train_path(df, layer_index)
    if "layer" in df.columns:
        try:
            return df.loc[df["layer"].astype(float).astype(int) == int(layer_index)]
        except Exception:
            return df.iloc[0:0]
    return df.iloc[0:0]


def _discover_layer_indices(df: pd.DataFrame) -> List[int]:
    """Discover available layer indices from a probe CSV DataFrame.

    - If train_dataset_path exists, parse 'layer_<idx>_train' occurrences (idx is zero-indexed)
    - Else, if 'layer' column exists, use its unique integer values
    Returns sorted unique indices.
    """
    if df is None or df.empty:
        return []
    indices: List[int] = []
    if "train_dataset_path" in df.columns:
        paths = df["train_dataset_path"].dropna().astype(str).tolist()
        for p in paths:
            m = re.search(r"layer_(\d+)_train", p)
            if m:
                try:
                    indices.append(int(m.group(1)))
                except Exception:
                    continue
    elif "layer" in df.columns:
        try:
            indices = [int(v) for v in pd.unique(df["layer"].dropna().astype(float).astype(int))]
        except Exception:
            indices = []
    return sorted(list(set(indices)))


def _choose_test_metric(df: pd.DataFrame, metric_priority: Optional[List[Tuple[str, str]]] = None) -> Optional[Tuple[str, str]]:
    """Choose a test-split metric column and its direction from a DataFrame.

    metric_priority: list of (column_name, direction) candidates.
    Returns (column_name, direction) or None if none found.
    """
    if df is None or df.empty:
        return None
    candidates = metric_priority or TEST_METRIC_PRIORITY_DEFAULT
    cols = set(df.columns)
    for name, direction in candidates:
        if name in cols:
            return name, direction
    return None


def get_probe_stats_for_layer(file_path: str, layer_index: int) -> Tuple[List[float], Optional[float]]:
    """Return (per-dataset |rho| list, mean |rho|) for the best avg_spearman row in a layer.

    - Reads a closed_form probe CSV
    - Filters by train_dataset_path for the provided layer index
    - Selects the row with maximum avg_spearman
    - Parses per_dataset_spearman and returns absolute values and their mean
    """
    if not os.path.exists(file_path):
        return [], None
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return [], None

    df_layer = filter_df_by_layer_train_path(df, layer_index)
    if df_layer is None or df_layer.empty:
        return [], None
    if "avg_spearman" not in df_layer.columns or "per_dataset_spearman" not in df_layer.columns:
        return [], None

    # Choose best row by avg_spearman
    best_row = df_layer.sort_values(by="avg_spearman", ascending=False).iloc[0]
    raw = str(best_row["per_dataset_spearman"]) if pd.notna(best_row["per_dataset_spearman"]) else "{}"

    # Parse JSON (handle single quotes fallback)
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(raw.replace("'", '"'))
        except Exception:
            parsed = {}

    per_values: List[float] = []
    for v in parsed.values():
        if v is None:
            continue
        try:
            per_values.append(abs(float(v)))
        except Exception:
            continue

    mean_abs = float(np.mean(per_values)) if len(per_values) > 0 else None
    return per_values, mean_abs


def build_probing_data_for_layer(
    probe_dir: str,
    models: List[str],
    conceptual_layer: int,
) -> Dict[str, Dict[str, object]]:
    """Build probing data dict for plotting using unified selection.

    Returns mapping model_display_name -> { 'avg_spearman', 'individual_values', 'steps' }.
    - avg_spearman is the mean of per-dataset |rho|
    - individual_values are the per-dataset |rho| values
    - steps is int or None (0 for original when plotting)
    """
    layer_index = conceptual_layer - 1
    results: Dict[str, Dict[str, object]] = {}

    for model in models:
        suffix = model_to_probe_suffix(model)
        csv_path = os.path.join(probe_dir, f"probe_results_{suffix}.csv")
        per_vals, mean_abs = get_probe_stats_for_layer(csv_path, layer_index)
        if mean_abs is None:
            # Skip models without data for this layer
            continue
        steps = extract_steps_from_model_dir(model)
        model_key = "original" if steps is None else f"{steps}steps"
        results[model_key] = {
            "avg_spearman": float(mean_abs),
            "individual_values": per_vals,
            "steps": steps,
        }
    return results


def prepare_plot_arrays_from_probing_data(
    probing_data: Dict[str, Dict[str, object]]
) -> Tuple[List[str], List[str], List[float], List[List[float]]]:
    """Prepare arrays for plotting from probing_data structure.

    Returns (models_in_order, step_labels, means, points_per_model)
    where step_labels are strings like "0", "100", ... with original shown as "0".
    """
    all_models = list(probing_data.keys())
    base_models = [m for m in all_models if m == "original"]
    ft_models = [m for m in all_models if m != "original"]
    ft_models = sorted(ft_models, key=lambda name: probing_data[name]["steps"] or 0)  # type: ignore[index]
    models = base_models + ft_models

    step_labels: List[str] = []
    means: List[float] = []
    points: List[List[float]] = []
    for m in models:
        steps = probing_data[m]["steps"]  # type: ignore[index]
        step_labels.append("0" if steps is None else str(steps))
        means.append(float(probing_data[m]["avg_spearman"]))  # type: ignore[index]
        points.append(list(probing_data[m]["individual_values"]))  # type: ignore[index]
    return models, step_labels, means, points


def get_best_train_rmse_layer_stats(file_path: str, layer_indices: list) -> tuple:
    """Find the layer with best (lowest) train_rmse and return its Spearman stats.
    
    Returns: (best_layer_index, per_dataset_abs_rho_list, mean_abs_rho)
    """
    if not os.path.exists(file_path):
        return None, [], None
    
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None, [], None
    
    best_train_rmse = None
    best_layer_idx = None
    best_per_values = []
    best_mean_abs = None
    
    for layer_idx in layer_indices:
        df_layer = filter_df_by_layer_train_path(df, layer_idx)
        if df_layer is None or df_layer.empty:
            continue
        if ("train_rmse" not in df_layer.columns or 
            "avg_spearman" not in df_layer.columns or 
            "per_dataset_spearman" not in df_layer.columns):
            continue
        
        # Find row with best train_rmse (lowest) in this layer
        best_row = df_layer.sort_values(by="train_rmse", ascending=True).iloc[0]
        train_rmse = best_row["train_rmse"]
        
        if pd.isna(train_rmse):
            continue
            
        # Check if this is the best train_rmse so far
        if best_train_rmse is None or train_rmse < best_train_rmse:
            best_train_rmse = train_rmse
            best_layer_idx = layer_idx
            
            # Parse the per_dataset_spearman for this row
            raw = str(best_row["per_dataset_spearman"]) if pd.notna(best_row["per_dataset_spearman"]) else "{}"
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = json.loads(raw.replace("'", '"'))
                except Exception:
                    parsed = {}
            
            per_values = []
            for v in parsed.values():
                if v is None:
                    continue
                try:
                    per_values.append(abs(float(v)))
                except Exception:
                    continue
            
            best_per_values = per_values
            best_mean_abs = float(np.mean(per_values)) if len(per_values) > 0 else None
    
    return best_layer_idx, best_per_values, best_mean_abs


def get_best_test_metric_layer_stats(
    file_path: str,
    metric_priority: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Optional[int], List[float], Optional[float], Optional[str], Optional[float]]:
    """Find the layer with best test performance and return its Spearman stats.

    Selection strategy:
    - Detect available layers in the CSV
    - Choose a test metric (per metric_priority, default TEST_METRIC_PRIORITY_DEFAULT)
    - For each layer, select the row with best value for that metric within the layer
    - Choose the overall best layer based on that metric

    Returns:
      (best_layer_index, per_dataset_abs_rho_list, mean_abs_rho, metric_name, metric_value)
    """
    if not os.path.exists(file_path):
        return None, [], None, None, None
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None, [], None, None, None

    # Decide which test metric to use
    metric_choice = _choose_test_metric(df, metric_priority)
    if metric_choice is None:
        return None, [], None, None, None
    metric_name, direction = metric_choice
    ascending = True if direction == "min" else False

    # Discover layer indices
    layer_indices = _discover_layer_indices(df)
    if not layer_indices:
        return None, [], None, metric_name, None

    best_layer_idx: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_per_values: List[float] = []
    best_mean_abs: Optional[float] = None

    for layer_idx in layer_indices:
        df_layer = _filter_df_by_layer_general(df, layer_idx)
        if df_layer is None or df_layer.empty or metric_name not in df_layer.columns:
            continue
        # Drop rows without the metric
        df_layer = df_layer.loc[pd.notna(df_layer[metric_name])]
        if df_layer.empty:
            continue

        # Pick best row in this layer for the chosen metric
        best_row = df_layer.sort_values(by=metric_name, ascending=ascending).iloc[0]
        row_metric_value = best_row[metric_name]
        try:
            row_metric_value_f = float(row_metric_value)
        except Exception:
            continue

        # Compare with global best
        if best_metric_value is None:
            is_better = True
        else:
            is_better = row_metric_value_f < best_metric_value if ascending else row_metric_value_f > best_metric_value
        if not is_better:
            continue

        best_layer_idx = layer_idx
        best_metric_value = row_metric_value_f

        # Parse per_dataset_spearman from the same row
        raw = str(best_row.get("per_dataset_spearman", "{}"))
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = json.loads(raw.replace("'", '"'))
            except Exception:
                parsed = {}
        per_values: List[float] = []
        for v in parsed.values():
            if v is None:
                continue
            try:
                per_values.append(abs(float(v)))
            except Exception:
                continue
        best_per_values = per_values
        best_mean_abs = float(np.mean(per_values)) if len(per_values) > 0 else None

    return best_layer_idx, best_per_values, best_mean_abs, metric_name, best_metric_value


def find_best_layers_all_models_by_test(
    zero_shot_base_dir: str = ZERO_SHOT_BASE_DIR_DEFAULT,
    probe_dir: str = PROBE_RESULTS_CLOSED_FORM_DIR_DEFAULT,
    metric_priority: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Compute best layer (by test metric) for base and all checkpoints.

    Returns mapping model_display_name -> {
        'best_layer_index': int | None,
        'best_conceptual_layer': int | None,
        'metric_name': str | None,
        'metric_value': float | None,
        'avg_spearman': float | None,              # mean of per-dataset |rho|
        'individual_values': List[float],          # per-dataset |rho|
        'steps': int | None,
    }
    """
    results: Dict[str, Dict[str, object]] = {}
    models = list_zero_shot_models(zero_shot_base_dir)
    for model in models:
        suffix = model_to_probe_suffix(model)
        csv_path = os.path.join(probe_dir, f"probe_results_{suffix}.csv")
        best_layer_idx, per_vals, mean_abs, metric_name, metric_value = get_best_test_metric_layer_stats(
            csv_path, metric_priority
        )
        # Skip if nothing found
        if best_layer_idx is None and metric_name is None:
            continue
        steps = extract_steps_from_model_dir(model)
        model_key = "original" if steps is None else f"{steps}steps"
        # Conceptual layer is 1-based when inferred from train_dataset_path indices
        conceptual_layer = best_layer_idx + 1 if best_layer_idx is not None else None
        results[model_key] = {
            "best_layer_index": best_layer_idx,
            "best_conceptual_layer": conceptual_layer,
            "metric_name": metric_name,
            "metric_value": float(metric_value) if metric_value is not None else None,
            "avg_spearman": float(mean_abs) if mean_abs is not None else None,
            "individual_values": per_vals,
            "steps": steps,
        }
    return results