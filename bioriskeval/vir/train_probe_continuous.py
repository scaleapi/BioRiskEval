import torch
import numpy as np
import h5py
import argparse
import wandb
import os
import random
import csv
from pathlib import Path
from typing import Tuple, cast, Optional

"""
This script trains a linear regression probe (single linear layer) on a dataset
with continuous labels.
"""

def parse_str_to_bool(value: str) -> bool:
    """Parse common string representations of truthy/falsey to bool."""
    return value.lower() in ("1", "true", "t", "yes", "y")

def read_probe_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, 'r') as f:
        ds_seq = cast(h5py.Dataset, f['sequences'])
        ds_rep = cast(h5py.Dataset, f['representations'])
        ds_lab = cast(h5py.Dataset, f['labels'])
        sequences: np.ndarray = ds_seq[:]
        representations: np.ndarray = ds_rep[:]
        # Note: original logic scales when '30' in path; keep for parity
        # if '30' in dataset_path:
        #     print("Scaling representations by 1e-10 (dividing by 1e10) for layer 30 files")
        #     representations = representations / 1e10
        labels: np.ndarray = ds_lab[:]
    return sequences, representations, labels


def solve_linear_probe(representations: np.ndarray, labels: np.ndarray, args: Optional[argparse.Namespace] = None) -> torch.nn.Module:
    """Solve linear regression probe using closed-form solution (normal equation).
    
    This computes the optimal weights directly without iterative training:
    w = (X^T X)^(-1) X^T y
    
    Parameters
    ----------
    representations : np.ndarray
        Array of latent representations with shape (N, D).
    labels : np.ndarray
        Continuous labels with shape (N,).
    args : argparse.Namespace, optional
        Command-line arguments (mainly for wandb logging).
    
    Returns
    -------
    torch.nn.Module
        Linear probe with optimal weights set.
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    x_tensor = torch.tensor(representations, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
    
    # Add bias term (column of ones) to X
    N, D = x_tensor.shape
    x_with_bias = torch.cat([x_tensor, torch.ones(N, 1, device=device)], dim=1)  # (N, D+1)
    
    # Solve using pseudoinverse: w = X^+ y
    # This is more numerically stable than (X^T X)^(-1) X^T y
    try:
        # Compute pseudoinverse
        x_pinv = torch.linalg.pinv(x_with_bias)  # (D+1, N)
        optimal_weights = x_pinv @ y_tensor  # (D+1, 1)
        
        # Split weights and bias
        w = optimal_weights[:-1, 0]  # (D,)
        b = optimal_weights[-1, 0]   # scalar
        
    except Exception as e:
        print(f"Warning: Pseudoinverse failed ({e}), falling back to lstsq solution")
        # Fallback to least squares solution
        solution = torch.linalg.lstsq(x_with_bias, y_tensor, rcond=None)
        optimal_weights = solution.solution  # (D+1, 1)
        w = optimal_weights[:-1, 0]
        b = optimal_weights[-1, 0]
    
    # Create probe and set optimal weights
    probe = torch.nn.Linear(D, 1).to(device)
    with torch.no_grad():
        probe.weight.copy_(w.unsqueeze(0))  # (1, D)
        probe.bias.copy_(b.unsqueeze(0))    # (1,)
    
    # Compute training loss for logging
    with torch.no_grad():
        preds = probe(x_tensor)
        mse_loss = torch.nn.functional.mse_loss(preds, y_tensor).item()
        mae_loss = torch.nn.functional.l1_loss(preds, y_tensor).item()
    
    print(f"Closed-form solution found:")
    print(f"  Training MSE: {mse_loss:.6f}")
    print(f"  Training MAE: {mae_loss:.6f}")
    
    # Log to wandb if enabled
    if args and hasattr(args, 'wandb') and args.wandb:
        wandb.log({
            "model/input_dim": D,
            "model/output_dim": 1,
            "dataset/num_samples": N,
            "labels/mean": float(labels.mean()),
            "labels/std": float(labels.std()),
            "labels/min": float(labels.min()),
            "labels/max": float(labels.max()),
            "train/method": "closed_form",
            "train/final_mse": mse_loss,
            "train/final_mae": mae_loss,
        })
    
    return probe


def train_linear_probe(representations: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> torch.nn.Module:
    """Train a simple linear regression probe using PyTorch.

    Parameters
    ----------
    representations : np.ndarray
        Array of latent representations with shape (N, D). D is expected to be 4096.
    labels : np.ndarray
        Continuous labels with shape (N,).
    args : argparse.Namespace
        Command-line arguments containing hyper-parameters (batch_size, num_steps/num_epochs, learning_rate).
    """
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert numpy arrays to torch tensors
    x_tensor = torch.tensor(representations, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

    # Build DataLoader
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Define a single linear layer as the probe
    probe = torch.nn.Linear(x_tensor.shape[1], 1).to(device)

    # Select regression loss
    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "mae":
        criterion = torch.nn.L1Loss()
    elif args.loss == "huber":
        # Use SmoothL1Loss as Huber with configurable beta/delta
        criterion = torch.nn.SmoothL1Loss(beta=args.huber_delta)
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)

    # Log model architecture and dataset info to wandb
    if args.wandb:
        wandb.log({
            "model/input_dim": x_tensor.shape[1],
            "model/output_dim": 1,
            "dataset/num_samples": len(dataset),
            "labels/mean": float(labels.mean()),
            "labels/std": float(labels.std()),
            "labels/min": float(labels.min()),
            "labels/max": float(labels.max()),
            "train/loss_type": args.loss,
        })

    probe.train()
    
    # Choose between epoch-based or step-based training
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        # Epoch-based training
        print(f"Training for {args.num_epochs} epochs...")
        total_batches = 0
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds = probe(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                # Detach before converting to Python float to avoid warnings
                epoch_loss += loss.detach().item()
                num_batches += 1
                total_batches += 1
            # Print and log epoch statistics
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.6f}")
            
            if args.wandb:
                wandb.log({
                    "train/epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/total_batches": total_batches,
                })
    else:
        # Step-based training (original behavior)
        print(f"Training for {args.num_steps} steps...")
        global_step = 0
        while global_step < args.num_steps:
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds = probe(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                if global_step % 10 == 0:
                    print(
                        f"Step {global_step}/{args.num_steps} | Loss: {loss.item():.6f}")
                    
                    if args.wandb:
                        wandb.log({
                            "train/step": global_step,
                            "train/loss": loss.detach().item(),
                        })
                            
                global_step += 1
                if global_step >= args.num_steps:
                    break

    print("Training complete.")
    return probe


def evaluate_probe(probe, test_representations, test_labels, args=None):
    """Evaluate the trained regression probe on a held-out set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(test_representations, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(test_labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
        preds = probe(x_tensor)

        # Compute regression metrics
        errors = preds - y_tensor
        mse = torch.mean(errors ** 2).item()
        rmse = float(np.sqrt(mse))
        mae = torch.mean(torch.abs(errors)).item()

        y_true_np = y_tensor.squeeze(1).detach().cpu().numpy()
        y_pred_np = preds.squeeze(1).detach().cpu().numpy()
        y_true_mean = float(y_true_np.mean())
        ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
        ss_tot = float(np.sum((y_true_np - y_true_mean) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        # Pearson correlation
        if np.std(y_true_np) > 0 and np.std(y_pred_np) > 0:
            pearson = float(np.corrcoef(y_true_np, y_pred_np)[0, 1])
        else:
            pearson = 0.0
        
        # Log test metrics to wandb
        if args and args.wandb:
            wandb.log({
                "test/mse": mse,
                "test/rmse": rmse,
                "test/mae": mae,
                "test/r2": r2,
                "test/pearson": pearson,
            })
    
    return rmse, mae, r2, pearson





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets/dms_probe_dataset_layer_26_train.h5")
    parser.add_argument("--test_dataset_path", type=str, default="/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets/dms_probe_dataset_layer_26_test.h5")
    
    # Training duration options (mutually exclusive)
    training_group = parser.add_mutually_exclusive_group(required=False)
    training_group.add_argument("--num_steps", type=int, default=100, help="Number of training steps (default: 100)")
    training_group.add_argument("--num_epochs", type=int, help="Number of training epochs (alternative to --num_steps)")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber"], help="Regression loss to use")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Delta parameter for Huber/SmoothL1 loss when --loss huber is selected")
    parser.add_argument("--shuffle_labels", type=str, default="False", help="Shuffle labels, for ablation study (accepts true/false)")
    parser.add_argument("--use_closed_form", action="store_true", help="Use closed-form solution instead of iterative training")
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="virulence-probe-continuous")
    parser.add_argument("--output_csv", type=str, default="probe_results_continuous.csv")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features")
    args = parser.parse_args()
    
    # Only require num_steps/num_epochs for iterative training
    if not args.use_closed_form:
        if args.num_steps is None and args.num_epochs is None:
            raise ValueError("Either --num_steps or --num_epochs must be specified for iterative training")
    
    # Initialize wandb
    if args.wandb:
        # Get project name from environment variable or use default
        wandb_project = args.wandb_project
        if args.use_closed_form:
            wandb_name = f"closedform_{args.train_dataset_path.split('/')[-1].replace('.h5', '')}_test_{args.test_dataset_path.split('/')[-1].replace('.h5', '')}_shuffle_labels_{args.shuffle_labels}"
        else:
            wandb_name = f"train_{args.train_dataset_path.split('/')[-1].replace('.h5', '')}_test_{args.test_dataset_path.split('/')[-1].replace('.h5', '')}_lr_{args.learning_rate}_bs_{args.batch_size}_steps_{args.num_steps}_epochs_{args.num_epochs}_loss_{args.loss}_shuffle_labels_{args.shuffle_labels}"
        # Create config dict for wandb
        if "evo2_7b_1m" in args.train_dataset_path or "evo2_40b_1m" in args.train_dataset_path:
            checkpoint_step = 0
        else:
            checkpoint_step = int(args.train_dataset_path.split("/")[-1].replace(".h5", "").split("=")[2].split("-")[0]) + 1
        config = {
            "shuffle_labels": args.shuffle_labels,
            "train_dataset_group": args.train_dataset_path.split("/")[-2],
            "layer": int(args.train_dataset_path.split("layer")[1].split("_")[1]),
            "checkpoint_step": checkpoint_step,
            "method": "closed_form" if args.use_closed_form else "iterative",
        }
        
        # Add training-specific config for iterative method
        if not args.use_closed_form:
            config["batch_size"] = args.batch_size
            config["learning_rate"] = args.learning_rate
            config["loss"] = args.loss
            # Add training duration to config
            if args.num_epochs is not None:
                config["num_epochs"] = args.num_epochs
                config["training_mode"] = "epochs"
            else:
                config["num_steps"] = args.num_steps
                config["training_mode"] = "steps"
            if args.loss == "huber":
                config["huber_delta"] = args.huber_delta
        
        wandb.init(
            project=wandb_project,
            config=config,
            name=wandb_name,
        )
        if wandb.run is not None:
            print(f"🚀 Wandb run initialized: {wandb.run.name}")
        else:
            print("🚀 Wandb run initialized")
    


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    

    train_sequences, train_representations, train_labels = read_probe_dataset(args.train_dataset_path)
    test_sequences, test_representations, test_labels = read_probe_dataset(args.test_dataset_path)
    # Log statistics of representation magnitudes for train/test sets
    if args.wandb:
        train_representations_magnitude = np.linalg.norm(train_representations, axis=1)
        test_representations_magnitude = np.linalg.norm(test_representations, axis=1)
        train_mag_mean = float(np.mean(train_representations_magnitude))
        test_mag_mean = float(np.mean(test_representations_magnitude))
        combined_mag_mean = float(np.mean(np.concatenate([
            train_representations_magnitude,
            test_representations_magnitude
        ], axis=0)))
        wandb.log({
            "magnitude/train_mean": train_mag_mean,
            "magnitude/test_mean": test_mag_mean,
            "magnitude/combined_mean": combined_mag_mean,
        })
    
    # Optionally shuffle training labels for ablation
    if args.shuffle_labels == "True":
        print("Shuffling training labels")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_labels)
    # Optional feature standardization
    if args.normalize_features:
        # Normalize each vector to have L2 norm = 1
        print("Normalizing features to unit vectors...")
        
        # # Compute L2 norms for each vector (row) in train_representations
        train_norms = np.linalg.norm(train_representations, axis=1, keepdims=True)
        # Avoid division by zero - replace zero norms with 1
        train_norms = np.where(train_norms == 0, 1, train_norms)
        # Normalize train representations
        train_representations = train_representations / train_norms
        
        # Compute L2 norms for each vector (row) in test_representations
        test_norms = np.linalg.norm(test_representations, axis=1, keepdims=True)
        # Avoid division by zero - replace zero norms with 1
        test_norms = np.where(test_norms == 0, 1, test_norms)
        # Normalize test representations
        test_representations = test_representations / test_norms

        
        # Log normalization statistics
        if args.wandb:
            # Verify normalization worked - compute norms after normalization
            train_norms_after = np.linalg.norm(train_representations, axis=1)
            test_norms_after = np.linalg.norm(test_representations, axis=1)
            
            wandb.log({
                "normalize_features": True,
                "magnitude/train_norm_mean_after": float(np.mean(train_norms_after)),
                "magnitude/train_norm_std_after": float(np.std(train_norms_after)),
                "magnitude/test_norm_mean_after": float(np.mean(test_norms_after)),
                "magnitude/test_norm_std_after": float(np.std(test_norms_after)),
            })
        print(f"Normalized {len(train_representations)} training vectors and {len(test_representations)} test vectors to unit norm")
    else:
        if args.wandb:
            wandb.log({
                "normalize_features": False,
            })
    
    # Choose between closed-form solution or iterative training
    if args.use_closed_form:
        print("Using closed-form solution for linear regression...")
        probe = solve_linear_probe(train_representations, train_labels, args)
    else:
        print("Using iterative training for linear regression...")
        probe = train_linear_probe(train_representations, train_labels, args)

    rmse, mae, r2, pearson = evaluate_probe(probe, test_representations, test_labels, args)
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R2: {r2:.6f}")
    print(f"Test Pearson: {pearson:.6f}")
    # Finish wandb run
    if args.wandb:
        wandb.finish()
        

    train_path_name = "/".join(args.train_dataset_path.split("/")[-2:]).replace(".h5", "")
    test_path_name = "/".join(args.test_dataset_path.split("/")[-2:]).replace(".h5", "")
    # Append results to CSV, create file with header if it does not exist
    result_file = Path(args.output_csv)
    file_exists = result_file.exists()
    with result_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["train_dataset_path", "test_dataset_path", "method", "probing_layer","learning_rate", "batch_size", "num_steps", "num_epochs", "loss", "rmse", "mae", "r2", "pearson", "shuffle_labels"])
        
        method = "closed_form" if args.use_closed_form else "iterative"
        learning_rate = "N/A" if args.use_closed_form else args.learning_rate
        batch_size = "N/A" if args.use_closed_form else args.batch_size
        num_steps = "N/A" if args.use_closed_form else args.num_steps
        num_epochs = "N/A" if args.use_closed_form else args.num_epochs
        loss = "N/A" if args.use_closed_form else args.loss
        # Extract the value immediately after "layer" in the filename
        filename = args.train_dataset_path.split("/")[-1].replace(".h5", "")

        probing_layer = args.train_dataset_path.split("layer")[1].split("_")[1]
        
        writer.writerow([train_path_name, test_path_name, method, probing_layer, learning_rate, batch_size, num_steps, num_epochs, loss, f"{rmse:.6f}", f"{mae:.6f}", f"{r2:.6f}", f"{pearson:.6f}", args.shuffle_labels])
    