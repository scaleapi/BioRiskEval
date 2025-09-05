#!/usr/bin/env python3
"""Sweep probe experiments with different parameters across multiple GPUs.

Usage examples:
    # Linear continuous on GPU 0
    python sweep_probes_dms.py --task_type continuous --gpu 0

    # Binary (always linear) on GPU 1
    python sweep_probes_dms.py --task_type binary --gpu 1

    # Closed-form (continuous only)
    python sweep_probes_dms.py --task_type continuous --closed_form --gpu 0

    # Non-linear (continuous only)
    python sweep_probes_dms.py --task_type continuous --non_linear --hidden_dims 512,512 --activation gelu --gpu 0

    # Run on multiple GPUs
    python sweep_probes_dms.py --task_type continuous --gpu 0,1

    # Run multiple instances in parallel on different GPUs:
    python sweep_probes_dms.py --task_type continuous --gpu 0 &
    python sweep_probes_dms.py --task_type binary --gpu 1 &
    wait  # Wait for both to complete
"""

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# Define parameter ranges to sweep
# Default fine-tuning steps (0 = original model)
DEFAULT_FT_STEPS = [0, 100, 200, 500, 1000, 2000]
LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#EPOCHS = [30, 50]  # Different numbers of epochs
EPOCHS = [0]
BATCH_SIZES = [0]
LEARNING_RATES = [0]

# Mapping of fine-tuning steps to their sample sizes (0 = original model)
STEP_TO_SAMPLES = {
    0: "7b",  # Special case for original model
    100: 800,
    200: 1600,
    500: 4000,
    1000: 8000,
    2000: 16000
}

def get_dataset_paths(ft_steps, layer, model_size,args):
    """Get train and test dataset paths for a given model size, fine-tuning step and layer."""
    if model_size == "40b":
        # Only original model supported for 40b unless additional FT directories are added
        model_dir = "nemo2_evo2_40b_1m"
        if ft_steps != 0:
            raise ValueError("40b sweep currently supports only original model (ft_steps=0)")
    else:
        # 7b model family
        if ft_steps == 0:
            model_dir = "nemo2_evo2_7b_1m"
        else:
            samples = STEP_TO_SAMPLES[ft_steps]
            model_dir = f"evo2_7b_1m_{ft_steps}_ncbi_virus_human_host_full_species_samples={samples}"

    # Construct full paths
    train_path = os.path.join(args.datasets_root, model_dir, f"dms_probe_dataset_layer_{layer}_train.h5")
    test_path = os.path.join(args.datasets_root, model_dir, f"dms_probe_dataset_layer_{layer}_val.h5")

    return train_path, test_path, model_dir

def run_probe(train_path, test_path, epochs, batch_size, learning_rate, task_type, loss_function, results_file, *, closed_form=False, non_linear=False, hidden_dims=None, activation=None, dropout=None,save_probe_path=None):
    """Run a single probe experiment with the given parameters."""
    cmd = [
        "python", "/workspaces/BioRiskEval/bioriskeval/mut/train_dms_probe.py",
        "--train_dataset_path", train_path,
        "--test_dataset_path", test_path,
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--task_type", task_type,
        "--loss", loss_function,
        "--results_file", results_file    ]
    if closed_form:
        cmd.append("--closed_form")
    if non_linear:
        cmd.append("--non_linear")
        if hidden_dims:
            cmd.extend(["--hidden_dims", hidden_dims])
        if activation:
            cmd.extend(["--activation", activation])
        if dropout is not None:
            cmd.extend(["--dropout", str(dropout)])
    if save_probe_path:
        cmd.extend(["--save_probe_path", save_probe_path])
    
    print("\nRunning probe with parameters:")
    print(f"  Train dataset: {train_path}")
    print(f"  Test dataset: {test_path}")
    print(f"  Task type: {task_type}")
    print(f"  Loss function: {loss_function}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Results file: {results_file}")
    
    try:
        # Pass current environment (including CUDA_VISIBLE_DEVICES) to subprocess
        subprocess.run(cmd, check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        print(f"Error running probe: {e}")
        return False
    return True


# Ensure local imports work when running this script from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


from train_dms_probe import (
    read_probe_dataset,
    solve_closed_form_probe,
    save_probe_artifacts,
    select_best_row,
)



def _infer_suffix(csv_path: Path) -> str:
    stem = csv_path.stem
    m = re.match(r"probe_results_(.+)", stem)
    return m.group(1) if m else "7b"


def _pick_model_subdir(datasets_root: Path, suffix: str) -> Optional[Path]:
    if not datasets_root.exists():
        return None
    subdirs = [p for p in datasets_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Handle explicit fine-tune step suffix like "1000steps"
    if suffix not in ("7b", "40b"):
        m = re.match(r"(\d+)steps", suffix)
        if m:
            steps = m.group(1)
            preferred = [p for p in subdirs if f"_1m_{steps}_" in p.name]
            if preferred:
                return sorted(preferred)[0]
            fallback = [p for p in subdirs if f"_{steps}_" in p.name]
            if fallback:
                return sorted(fallback)[0]
    # Prefer by model family when suffix indicates size explicitly
    if suffix == "7b":
        seven_b = [p for p in subdirs if "7b" in p.name]
        if seven_b:
            return sorted(seven_b)[0]
    if suffix == "40b":
        forty_b = [p for p in subdirs if "40b" in p.name]
        if forty_b:
            return sorted(forty_b)[0]
    # Fallbacks
    nemo_like = [p for p in subdirs if "nemo2" in p.name]
    if nemo_like:
        # Prefer 7b over 40b if both exist and suffix is ambiguous
        seven_b = [p for p in nemo_like if "7b" in p.name]
        if seven_b:
            return sorted(seven_b)[0]
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


def _save_closed_form_best_from_results(results_dir: Path, datasets_root: Path, probes_root: Path, *, selection: str, seed: int = 42) -> None:
    """Select best rows from results and save closed-form probes accordingly.

    selection: "train_rmse" or "val_spearman" determines metric and subfolder.
    """
    if select_best_row is None or read_probe_dataset is None or solve_closed_form_probe is None or save_probe_artifacts is None:
        raise ImportError("Required utilities could not be imported. Please run this script from the ft-attack/probe directory or ensure PYTHONPATH includes it.")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    csv_files = sorted(results_dir.glob("probe_results_*.csv"))
    if not csv_files:
        print(f"No probe_results_*.csv files found in {results_dir}")
        return

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
        if train_token.endswith(".h5"):
            train_token = train_token[:-3]
        layer_inferred = _parse_layer_from_token(train_token)

        train_path = model_subdir / f"{train_token}.h5"
        if not train_path.exists():
            print(f"Skipping {csv_path.name}: train dataset not found: {train_path}")
            continue

        # Load training data and solve closed-form
        _, train_repr, train_labels, _ = read_probe_dataset(str(train_path))
        probe, final_train_mse, train_rmse = solve_closed_form_probe(train_repr, train_labels, args=None)

        # Build metadata for saving
        training_mode = "closed_form_train_rmse" if selection == "train_rmse" else "closed_form"
        selection_strategy = "train_rmse" if selection == "train_rmse" else "val_spearman"
        metadata = {
            "task_type": "continuous",
            "training_mode": training_mode,
            "selection_strategy": selection_strategy,
            "loss_function": "mse",
            "seed": int(seed),
            "batch_size": 0,
            "learning_rate": 0.0,
            "num_steps": None,
            "num_epochs": None,
            "layer": layer_inferred,
            "train_dataset": train_path.name,
            "test_dataset": None,
            "wandb_run": "no_wandb",
            "metrics": {
                "final_train_loss": float(final_train_mse),
                "train_rmse": float(train_rmse),
            },
            "non_linear": False,
            "hidden_dims": None,
            "activation": None,
            "dropout": None,
            "model_id": model_subdir.name,
        }

        save_probe_artifacts(probe, str(probes_root), metadata)
        print(f"Saved {selection_strategy}-selected probe for {csv_path.name}: layer={layer_inferred}")

        # Prune non-best probes in the model folder for this selection strategy
        subfolder = "closed_form_train_rmse" if selection == "train_rmse" else "closed_form_val_spearman"
        model_dir = probes_root / subfolder / model_subdir.name
        if model_dir.exists():
            for fp in model_dir.glob("*.*"):
                try:
                    name = fp.name
                    if (name.endswith('.pt') or name.endswith('.json')) and (layer_inferred is not None) and f"layer={layer_inferred}_" not in name:
                        fp.unlink()
                except Exception:
                    continue

def main():
    """Run sweep; optionally save best closed-form probes by selected metrics."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sweep probe experiments with different parameters")
    parser.add_argument(
        "--task_type", 
        type=str, 
        choices=["binary", "continuous"], 
        default="continuous",
        help="Type of task to run (binary or continuous)"
    )
    parser.add_argument(
        "--gpu", 
        type=str, 
        default=None,
        help="GPU device to use (e.g., '0', '1', '0,1' for multiple GPUs, or 'auto' to use all available). If not specified, uses default CUDA behavior."
    )
    parser.add_argument("--closed_form", action="store_true", help="Use closed-form solution (continuous only)")
    parser.add_argument("--non_linear", action="store_true", help="Use MLP probe (continuous only)")
    parser.add_argument("--hidden_dims", type=str, default="512", help="Comma-separated hidden layer sizes for MLP")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "tanh"], help="Activation for MLP")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for MLP")
    parser.add_argument("--save_probe_path", type=str, default=None, help="Path to save the probe (.pt/.pth or a directory). Saves a metadata JSON alongside.")
    parser.add_argument("--model_size", type=str, choices=["7b", "40b"], default="7b", help="Model size to sweep (7b or 40b). 40b supports original only.")
    parser.add_argument("--ft_steps", type=str, default=",".join(str(s) for s in DEFAULT_FT_STEPS), help="Comma-separated fine-tuning steps to evaluate (0 denotes original model). Default: 0,100,200,500,1000,2000")
    # Best-probe saving options
    parser.add_argument("--save_best_by_train_rmse", action="store_true", help="After sweep, save closed-form probes selected by best train RMSE")
    parser.add_argument("--save_best_by_val_spearman", action="store_true", help="After sweep, save closed-form probes selected by best mean-absolute val Spearman")
    parser.add_argument("--datasets_root", type=str, default="/workspaces/BioRiskEval/bioriskeval/mut/data/probe_datasets_stratified/k=500_seed=42", help="Root directory containing per-model probe datasets")
    parser.add_argument("--probes_root", type=str, default="/workspaces/BioRiskEval/bioriskeval/mut/saved_probes_stratified/k=500_seed=42", help="Root directory to save selected probes")
    parser.add_argument("--results_dir", type=str, default="/workspaces/BioRiskEval/bioriskeval/mut/probe_results_stratified/k=500_seed=42", help="Root directory to save selected probes")
    args = parser.parse_args()
    TASK_TYPE = args.task_type
    MODEL_SIZE = args.model_size

    # Set GPU environment if specified
    if args.gpu is not None:
        if args.gpu.lower() == 'auto':
            # Let CUDA use all available GPUs
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
        else:
            # Set specific GPU(s)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")

    # Validate mode compatibility and set loss functions
    if args.closed_form and args.non_linear:
        raise ValueError("--closed_form is not compatible with --non_linear")
    if TASK_TYPE == "binary":
        if args.closed_form or args.non_linear:
            raise ValueError("binary task supports only linear probe (no --closed_form or --non_linear)")
        LOSS_FUNCTIONS = ["bce"]
    else:  # continuous
        if args.closed_form:
            LOSS_FUNCTIONS = ["mse"]
        else:
            LOSS_FUNCTIONS = ["mse"]  # keep default; expand if needed
    # Alternative with more loss functions:
    # LOSS_FUNCTIONS = ["mse", "mae", "huber"] if TASK_TYPE == "continuous" else ["bce"]

    # Create results directory by mode for backward compatibility
    if TASK_TYPE == "binary":
        subfolder = "binary"
    else:
        if args.closed_form:
            subfolder = "closed_form"
        elif args.non_linear:
            subfolder = "nonlinear"
        else:
            subfolder = "continuous"
    task_results_dir = Path(args.results_dir) / subfolder
    task_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {TASK_TYPE} probe experiments")
    print(f"Results will be saved to: {task_results_dir}")
    print(f"Loss functions to sweep: {LOSS_FUNCTIONS}")

    # Select steps per model size (40b supports only original for now)
    try:
        parsed_steps = [int(s.strip()) for s in str(args.ft_steps).split(',') if s.strip() != ""]
    except Exception:
        parsed_steps = list(DEFAULT_FT_STEPS)
    steps_to_run = parsed_steps if MODEL_SIZE == "7b" else [0]

    # Run experiments for each fine-tuning step separately
    for ft_steps in steps_to_run:
        print(f"\n{'='*80}")
        if MODEL_SIZE == "40b":
            print(f"Running experiments for original 40b model ({TASK_TYPE})")
            model_name = "nemo2_evo2_40b_1m"
        else:
            if ft_steps == 0:
                print(f"Running experiments for original model ({TASK_TYPE})")
                model_name = "nemo2_evo2_7b_1m"
            else:
                print(f"Running experiments for fine-tuning steps: {ft_steps} ({TASK_TYPE})")
                samples = STEP_TO_SAMPLES[ft_steps]
                model_name = f"evo2_7b_1m_{ft_steps}_ncbi_virus_human_host_full_species_samples={samples}"
        print(f"{'='*80}")

        # Calculate total experiments for this fine-tuning step
        total_experiments = len(LAYERS) * len(EPOCHS) * len(BATCH_SIZES) * len(LEARNING_RATES) * len(LOSS_FUNCTIONS)
        print(f"Total number of experiments for {ft_steps} steps: {total_experiments}")

        experiment_count = 0

        # Run experiments for all parameter combinations including loss functions
        for layer, epochs, batch_size, lr, loss_func in itertools.product(
            LAYERS, EPOCHS, BATCH_SIZES, LEARNING_RATES, LOSS_FUNCTIONS
        ):
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}")
            print(f"Model: {model_name}")
            print(f"Layer: {layer}")

            try:
                train_path, test_path, _ = get_dataset_paths(ft_steps, layer, MODEL_SIZE, args)
            except ValueError:
                # Skip unsupported configurations (e.g., 40b with ft_steps>0)
                print(f"Skipping layer={layer} - unsupported configuration for model_size={MODEL_SIZE}, ft_steps={ft_steps}")
                continue
            print(train_path)

            # Skip if datasets don't exist
            if not (os.path.exists(train_path) and os.path.exists(test_path)):
                print(f"Skipping layer={layer} - datasets not found")
                continue

            # Create results file name with task type and loss function
            if MODEL_SIZE == "40b":
                results_file = task_results_dir / "probe_results_40b.csv"
            else:
                if ft_steps == 0:
                    results_file = task_results_dir / "probe_results_7b.csv"
                else:
                    results_file = task_results_dir / f"probe_results_{ft_steps}steps.csv"

            success = run_probe(
                train_path=train_path,
                test_path=test_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                task_type=TASK_TYPE,
                loss_function=loss_func,
                results_file=str(results_file),
                closed_form=bool(args.closed_form),
                non_linear=bool(args.non_linear),
                hidden_dims=args.hidden_dims if args.non_linear else None,
                activation=args.activation if args.non_linear else None,
                dropout=args.dropout if args.non_linear else None,
                save_probe_path=args.save_probe_path if args.save_probe_path else None,
            )

            if not success:
                print(f"Failed to run experiment {experiment_count}")

    # After sweep completes, optionally save best closed-form probes based on selections
    if TASK_TYPE == "continuous":
        results_dir = task_results_dir
        datasets_root = Path(args.datasets_root)
        probes_root = Path(args.probes_root)
        if args.save_best_by_train_rmse:
            print("\nSaving best closed-form probes selected by train RMSE...")
            _save_closed_form_best_from_results(results_dir, datasets_root, probes_root, selection="train_rmse", seed=42)
        if args.save_best_by_val_spearman:
            print("\nSaving best closed-form probes selected by mean-absolute val Spearman...")
            _save_closed_form_best_from_results(results_dir, datasets_root, probes_root, selection="val_spearman", seed=42)

if __name__ == "__main__":
    main()