import torch
import numpy as np
import pandas as pd
import h5py
import argparse
import wandb
import os
import random
import csv
from pathlib import Path
from typing import Tuple, cast
from sklearn.metrics import roc_auc_score

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
        if '30' in dataset_path:
            print("Scaling representations by 1e-10 (dividing by 1e10) for layer 30 files")
            representations = representations / 1e10
        labels: np.ndarray = ds_lab[:]
    return sequences, representations, labels


# def maybe_standardize_features(train_x: np.ndarray, test_x: np.ndarray, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
#     if not enabled:
#         return train_x, test_x
#     mean = train_x.mean(axis=0, keepdims=True)
#     std = train_x.std(axis=0, keepdims=True)
#     std = np.where(std < 1e-12, 1.0, std)
#     train_x_std = (train_x - mean) / std
#     test_x_std = (test_x - mean) / std
#     return train_x_std, test_x_std


def train_linear_probe(representations: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> torch.nn.Module:
    """Train a simple linear probe (logistic regression) using PyTorch.

    Parameters
    ----------
    representations : np.ndarray
        Array of latent representations with shape (N, D). D is expected to be 4096.
    labels : np.ndarray
        Binary labels with shape (N,).
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

    # Handle potential class imbalance (optional)
    if args.use_pos_weight:
        num_pos = float((labels == 1).sum())
        num_neg = float((labels == 0).sum())
        pos_weight_value = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)

    # Log model architecture and dataset info to wandb
    if args.wandb:
        wandb.log({
            "model/input_dim": x_tensor.shape[1],
            "model/output_dim": 1,
            "dataset/num_samples": len(dataset),
            "dataset/num_positive": labels.sum(),
            "dataset/num_negative": len(labels) - labels.sum(),
            "dataset/positive_ratio": labels.mean(),
        })

    probe.train()
    
    # Choose between epoch-based or step-based training
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        # Epoch-based training
        print(f"Training for {args.num_epochs} epochs...")
        total_batches = 0
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                logits = probe(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                # Compute accuracy for the current batch
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    acc = (preds == batch_y).float().mean().item()
                
                epoch_loss += loss.item()
                epoch_acc += acc
                num_batches += 1
                total_batches += 1
            # Print and log epoch statistics
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            
            if args.wandb:
                wandb.log({
                    "train/epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/accuracy": avg_acc,
                    "train/total_batches": total_batches,
                })
    else:
        # Step-based training (original behavior)
        print(f"Training for {args.num_steps} steps...")
        global_step = 0
        while global_step < args.num_steps:
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                logits = probe(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                if global_step % 10 == 0:
                    # Compute accuracy for the current batch
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        acc = (preds == batch_y).float().mean().item()
                        print(
                            f"Step {global_step}/{args.num_steps} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
                        
                        if args.wandb:
                            wandb.log({
                                "train/step": global_step,
                                "train/loss": loss.item(),
                                "train/accuracy": acc,
                            })
                            
                global_step += 1
                if global_step >= args.num_steps:
                    break

    print("Training complete.")
    return probe


def evaluate_probe(probe, test_representations, test_labels, args=None):
    """Evaluate the trained probe on a held-out set.

    Predictions are obtained by applying a sigmoid to the logits and using a
    0.5 threshold to map probabilities to the binary classes {0, 1}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(test_representations, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(test_labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
        logits = probe(x_tensor)
        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).float()
        test_acc = (preds == y_tensor).float().mean().item()

        # Calculate additional metrics
        test_loss = torch.nn.BCEWithLogitsLoss()(logits, y_tensor).item()

        # Calculate precision, recall, and F1
        tp = ((preds == 1) & (y_tensor == 1)).float().sum().item()
        fp = ((preds == 1) & (y_tensor == 0)).float().sum().item()
        tn = ((preds == 0) & (y_tensor == 0)).float().sum().item()
        fn = ((preds == 0) & (y_tensor == 1)).float().sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = roc_auc_score(test_labels, probs.squeeze(1).detach().cpu().numpy())
        
        # Log test metrics to wandb
        if args and args.wandb:
            wandb.log({
                "test/accuracy": test_acc,
                "test/loss": test_loss,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
                "test/true_positives": tp,
                "test/false_positives": fp,
                "test/true_negatives": tn,
                "test/false_negatives": fn,
                "test/auc": auc,
            })
    
    return test_acc, auc





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
    parser.add_argument("--use_pos_weight", action="store_true", help="Use positive class weighting for BCE loss")
    parser.add_argument("--shuffle_labels", type=str, default="False", help="Shuffle labels, for ablation study (accepts true/false)")
    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="virulence-probe")
    
    args = parser.parse_args()
    
    if args.num_steps is None and args.num_epochs is None:
        raise ValueError("Either --num_steps or --num_epochs must be specified")
    
    # Initialize wandb
    if args.wandb:
        # Get project name from environment variable or use default
        wandb_project = args.wandb_project
        wandb_name = f"train_{args.train_dataset_path.split('/')[-1].replace('.h5', '')}_test_{args.test_dataset_path.split('/')[-1].replace('.h5', '')}_lr_{args.learning_rate}_bs_{args.batch_size}_steps_{args.num_steps}_epochs_{args.num_epochs}_shuffle_labels_{args.shuffle_labels}"
        # Create config dict for wandb
        if "evo2_7b_1m" in args.train_dataset_path or "evo2_40b_1m" in args.train_dataset_path:
            checkpoint_step = 0
        else:
            checkpoint_step = int(args.train_dataset_path.split("/")[-1].replace(".h5", "").split("=")[2].split("-")[0]) + 1
        config = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "shuffle_labels": args.shuffle_labels,
            "train_dataset_group": args.train_dataset_path.split("/")[-2],
            "layer": args.train_dataset_path.split("/")[-1].replace(".h5", "").split("_")[-2],
            "checkpoint_step": checkpoint_step,
        }
        # Add training duration to config
        if args.num_epochs is not None:
            config["num_epochs"] = args.num_epochs
            config["training_mode"] = "epochs"
        else:
            config["num_steps"] = args.num_steps
            config["training_mode"] = "steps"
        
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
    
    # Optionally shuffle training labels for ablation
    if args.shuffle_labels == "True":
        print("Shuffling training labels")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_labels)
    # Optional feature standardization
    # train_representations, test_representations = maybe_standardize_features(
    #     train_representations, test_representations, enabled=args.standardize
    # )
    
    probe = train_linear_probe(train_representations, train_labels, args)

    test_acc, auc = evaluate_probe(probe, test_representations, test_labels, args)
    print(f"Test accuracy: {test_acc:.4f}")    
    print(f"Test AUC: {auc:.4f}")
    # Finish wandb run
    if args.wandb:
        wandb.finish()
        

    train_path_name = "/".join(args.train_dataset_path.split("/")[-2:]).replace(".h5", "")
    test_path_name = "/".join(args.test_dataset_path.split("/")[-2:]).replace(".h5", "")
    # Append results to CSV, create file with header if it does not exist
    result_file = Path("probe_results.csv")
    file_exists = result_file.exists()
    with result_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["train_dataset_path", "test_dataset_path","learning_rate", "batch_size", "num_steps", "test_acc", "auc", "shuffle_labels"])
        
        writer.writerow([train_path_name, test_path_name, args.learning_rate, args.batch_size, args.num_steps, f"{test_acc:.4f}", f"{auc:.4f}", args.shuffle_labels])
    