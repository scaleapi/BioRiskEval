#!/usr/bin/env python3
"""
Sweep probe experiments with different parameters across multiple GPUs.

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

import subprocess
import itertools
from pathlib import Path
import os
import argparse

# Configuration
BASE_DATA_DIR = "/workspaces/src/bionemo-framework/attack/data/eval_dataset/fitness/probe_datasets"
BASE_RESULTS_DIR = "probe_results"  # Base directory to store all results

# Define parameter ranges to sweep
FT_STEPS = [0, 100, 200, 500, 1000, 2000]  # Include 0 for original model
LAYERS = [22, 25]  # Example layers to test
EPOCHS = [30, 50]  # Different numbers of epochs
BATCH_SIZES = [128, 256]
LEARNING_RATES = [1e-3, 1e-4]

# Mapping of fine-tuning steps to their sample sizes (0 = original model)
STEP_TO_SAMPLES = {
    0: "original",  # Special case for original model
    100: 800,
    200: 1600,
    500: 4000,
    1000: 8000,
    2000: 16000
}

def get_dataset_paths(ft_steps, layer):
    """Get train and test dataset paths for a given fine-tuning step and layer."""
    if ft_steps == 0:
        # Special case for original model
        model_dir = "nemo2_evo2_7b_1m"
    else:
        # Construct the model directory name for fine-tuned models
        samples = STEP_TO_SAMPLES[ft_steps]
        model_dir = f"evo2_7b_1m_{ft_steps}_ncbi_virus_human_host_full_species_samples={samples}"
    
    # Construct full paths
    train_path = os.path.join(BASE_DATA_DIR, model_dir, f"dms_probe_dataset_layer_{layer}_train.h5")
    test_path = os.path.join(BASE_DATA_DIR, model_dir, f"dms_probe_dataset_layer_{layer}_test.h5")
    
    return train_path, test_path, model_dir

def run_probe(train_path, test_path, epochs, batch_size, learning_rate, task_type, loss_function, results_file, *, closed_form=False, non_linear=False, hidden_dims=None, activation=None, dropout=None):
    """Run a single probe experiment with the given parameters."""
    cmd = [
        "python", "probe/train_probe_dms.py",
        "--train_dataset_path", train_path,
        "--test_dataset_path", test_path,
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--task_type", task_type,
        "--loss", loss_function,
        "--results_file", results_file,
        "--wandb"  # Enable wandb logging
    ]
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
    
    print(f"\nRunning probe with parameters:")
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

def main():
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
    args = parser.parse_args()
    TASK_TYPE = args.task_type
    
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
    task_results_dir = Path(BASE_RESULTS_DIR) / subfolder
    task_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {TASK_TYPE} probe experiments")
    print(f"Results will be saved to: {task_results_dir}")
    print(f"Loss functions to sweep: {LOSS_FUNCTIONS}")
    
    # Run experiments for each fine-tuning step separately
    for ft_steps in FT_STEPS:
        print(f"\n{'='*80}")
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
            
            train_path, test_path, _ = get_dataset_paths(ft_steps, layer)
            
            # Skip if datasets don't exist
            if not (os.path.exists(train_path) and os.path.exists(test_path)):
                print(f"Skipping layer={layer} - datasets not found")
                continue
            
            # Create results file name with task type and loss function
            if ft_steps == 0:
                results_file = task_results_dir / f"probe_results_original.csv"
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
            )
            
            if not success:
                print(f"Failed to run experiment {experiment_count}")

if __name__ == "__main__":
    main() 