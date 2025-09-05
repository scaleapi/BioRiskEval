import torch
import numpy as np
import pandas as pd
import h5py
import argparse
import wandb
import os
import random
import csv
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from datetime import datetime
import re
from typing import cast, Tuple, Optional



def compute_mean_abs_per_dataset_spearman(raw: object) -> Optional[float]:
    if raw is None:
        return None
    # Normalize to string and parse JSON
    txt = str(raw)
    try:
        parsed = json.loads(txt)
    except Exception:
        try:
            parsed = json.loads(txt.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    values = []
    for v in parsed.values():
        if v is None:
            continue
        try:
            values.append(abs(float(v)))
        except Exception:
            continue
    if not values:
        return None
    return float(np.mean(values))

def select_best_row(df: pd.DataFrame, task_type: str, probe_pick: Optional[str] = None) -> pd.Series:
    """Select the best row from a results DataFrame.

    - Binary: maximize test_auc
    - Continuous:
      - probe_pick == 'train_rmse': minimize train_rmse
      - else 'val_spearman': strictly maximize mean(|per_dataset_spearman|)
        (no fallback). Requires per_dataset_spearman column with valid values.
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame provided to select_best_row")

    if task_type == "binary":
        if "test_auc" not in df.columns:
            raise ValueError("CSV missing 'test_auc' column for binary selection")
        return df.sort_values("test_auc", ascending=False).iloc[0]

    # Continuous
    pick_mode = (probe_pick or "val_spearman").lower()
    if pick_mode == "train_rmse":
        if "train_rmse" not in df.columns:
            raise ValueError("CSV missing 'train_rmse' column for probe_pick=train_rmse")
        return df.sort_values("train_rmse", ascending=True).iloc[0]

    # val_spearman: strictly use per-dataset mean abs
    df_local = df.copy()
    if "per_dataset_spearman" not in df_local.columns:
        raise ValueError("CSV missing 'per_dataset_spearman' required for mean-absolute selection")
    df_local["mean_abs_spearman"] = df_local["per_dataset_spearman"].apply(
        compute_mean_abs_per_dataset_spearman
    )
    if not df_local["mean_abs_spearman"].notna().any():
        raise ValueError("No valid per-dataset Spearman values to compute mean-absolute selection")
    return df_local.sort_values("mean_abs_spearman", ascending=False).iloc[0]




def read_probe_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        seq_ds = cast(h5py.Dataset, f['sequences'])
        rep_ds = cast(h5py.Dataset, f['representations'])
        lab_ds = cast(h5py.Dataset, f['labels'])

        sequences = seq_ds[:]
        representations = rep_ds[:]
        if '30' in str(dataset_path):
            print("Scaling representations by 1e10")
            representations = representations / 1e10
        labels = lab_ds[:]
        
        # Read DMS dataset identifiers if available
        dms_datasets = None
        if 'DMS_dataset' in f:
            dms_ds = cast(h5py.Dataset, f['DMS_dataset'])
            dms_datasets = dms_ds[:]
            # Convert bytes to strings if needed
            if hasattr(dms_datasets, 'dtype') and getattr(dms_datasets.dtype, 'kind', '') == 'S':
                dms_datasets = np.array([ds.decode('utf-8') for ds in dms_datasets])
    
    # Debug: Print dataset size info
    print(f"Loaded dataset from {dataset_path}:")
    print(f"  - Sequences shape: {sequences.shape}")
    print(f"  - Representations shape: {representations.shape}")
    print(f"  - Labels shape: {labels.shape}")
    if dms_datasets is not None:
        unique_datasets = np.unique(dms_datasets)
        print(f"  - DMS datasets: {len(unique_datasets)} unique datasets")
        print(f"  - Dataset distribution: {[(ds, np.sum(dms_datasets == ds)) for ds in unique_datasets[:5]]}")
    else:
        print(f"  - No DMS dataset identifiers found")
    
    return sequences, representations, labels, dms_datasets


def train_linear_probe(representations, labels, args):
    """Train a simple linear probe using PyTorch.

    Parameters
    ----------
    representations : np.ndarray
        Array of latent representations with shape (N, D). D is expected to be 4096.
    labels : np.ndarray
        Binary labels (for classification) or continuous labels (for regression) with shape (N,).
    args : argparse.Namespace
        Command-line arguments containing hyper-parameters (batch_size, num_steps/num_epochs, learning_rate, task_type).
        
    Returns
    -------
    probe : torch.nn.Linear
        The trained probe model.
    final_train_metric : float
        The final training metric (accuracy for binary, loss for continuous).
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

    # Build probe model: linear or optional MLP
    input_dim = x_tensor.shape[1]
    if getattr(args, 'non_linear', False):
        # Parse hidden dims
        hidden_dims_arg = getattr(args, 'hidden_dims', '512')
        if isinstance(hidden_dims_arg, str):
            hidden_dims = [int(h) for h in hidden_dims_arg.split(',') if h.strip()]
        elif isinstance(hidden_dims_arg, (list, tuple)):
            hidden_dims = [int(h) for h in hidden_dims_arg]
        else:
            hidden_dims = [512]

        activation_name = getattr(args, 'activation', 'relu').lower()
        if activation_name == 'relu':
            act_layer = torch.nn.ReLU
        elif activation_name == 'gelu':
            act_layer = torch.nn.GELU
        elif activation_name == 'tanh':
            act_layer = torch.nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        dropout_p = float(getattr(args, 'dropout', 0.0))

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(act_layer())
            if dropout_p > 0:
                layers.append(torch.nn.Dropout(p=dropout_p))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        probe = torch.nn.Sequential(*layers).to(device)
    else:
        # Define a single linear layer as the probe
        probe = torch.nn.Linear(input_dim, 1).to(device)

    # Select loss function based on task type
    task_type = getattr(args, 'task_type', 'binary')  # Default to binary for backward compatibility
    
    if task_type == 'binary':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task_type == 'continuous':
        # Select regression loss
        loss_type = getattr(args, 'loss', 'mse')  # Default to MSE
        if loss_type == "mse":
            criterion = torch.nn.MSELoss()
        elif loss_type == "mae":
            criterion = torch.nn.L1Loss()
        elif loss_type == "huber":
            # Use SmoothL1Loss as Huber with configurable beta/delta
            huber_delta = getattr(args, 'huber_delta', 1.0)
            criterion = torch.nn.SmoothL1Loss(beta=huber_delta)
        else:
            raise ValueError(f"Unsupported continuous loss: {loss_type}")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
        
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)

    # Log model architecture and dataset info to wandb
    if args.wandb:
        log_data = {
            "model/input_dim": input_dim,
            "model/output_dim": 1,
            "dataset/num_samples": len(dataset),
            "task_type": task_type,
        }
        # Log architecture details if MLP
        if getattr(args, 'non_linear', False):
            log_data.update({
                "model/arch": "mlp",
                "model/hidden_dims": str(hidden_dims),
                "model/activation": activation_name,
                "model/dropout": dropout_p,
            })
        
        if task_type == 'binary':
            log_data.update({
                "dataset/num_positive": labels.sum(),
                "dataset/num_negative": len(labels) - labels.sum(),
                "dataset/positive_ratio": labels.mean(),
            })
        elif task_type == 'continuous':
            log_data.update({
                "labels/mean": float(labels.mean()),
                "labels/std": float(labels.std()),
                "labels/min": float(labels.min()),
                "labels/max": float(labels.max()),
                "train/loss_type": getattr(args, 'loss', 'mse'),
            })
        
        wandb.log(log_data)

    probe.train()
    
    # Initialize final metric variable
    final_train_metric = 0.0
    
    # Choose between epoch-based or step-based training
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        # Epoch-based training
        print(f"Training for {args.num_epochs} epochs...")
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            epoch_metric = 0.0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                if task_type == 'binary':
                    logits = probe(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    # Compute accuracy for the current batch
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        metric = (preds == batch_y).float().mean().item()
                elif task_type == 'continuous':
                    preds = probe(batch_x)
                    loss = criterion(preds, batch_y)
                    metric = loss.item()  # Use loss as metric for continuous
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_metric += metric
                num_batches += 1

            # Print and log epoch statistics
            avg_loss = epoch_loss / num_batches
            avg_metric = epoch_metric / num_batches
            final_train_metric = avg_metric
            
            if task_type == 'binary':
                print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_metric:.4f}")
                if args.wandb:
                    wandb.log({
                        "train/epoch": epoch + 1,
                        "train/loss": avg_loss,
                        "train/accuracy": avg_metric,
                    })
            elif task_type == 'continuous':
                print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.6f}")
                if args.wandb:
                    wandb.log({
                        "train/epoch": epoch + 1,
                        "train/loss": avg_loss,
                    })
    else:
        # Step-based training (original behavior)
        print(f"Training for {args.num_steps} steps...")
        global_step = 0
        while global_step < args.num_steps:
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                if task_type == 'binary':
                    logits = probe(batch_x)
                    loss = criterion(logits, batch_y)
                elif task_type == 'continuous':
                    preds = probe(batch_x)
                    loss = criterion(preds, batch_y)

                loss.backward()
                optimizer.step()

                if global_step % 10 == 0:
                    if task_type == 'binary':
                        # Compute accuracy for the current batch
                        with torch.no_grad():
                            preds = (torch.sigmoid(logits) > 0.5).float()
                            metric = (preds == batch_y).float().mean().item()
                            final_train_metric = metric
                            print(f"Step {global_step}/{args.num_steps} | Loss: {loss.item():.4f} | Acc: {metric:.4f}")

                            if args.wandb:
                                wandb.log({
                                    "train/step": global_step,
                                    "train/loss": loss.item(),
                                    "train/accuracy": metric,
                                })
                    elif task_type == 'continuous':
                        final_train_metric = loss.item()
                        print(f"Step {global_step}/{args.num_steps} | Loss: {loss.item():.6f}")

                        if args.wandb:
                            wandb.log({
                                "train/step": global_step,
                                "train/loss": loss.item(),
                            })
                global_step += 1
                if global_step >= args.num_steps:
                    break

    print("Training complete.")
    return probe, final_train_metric


def solve_closed_form_probe(representations, labels, args=None):
    """Solve linear regression probe in closed form (least squares) without training.

    Uses pseudoinverse to find weights that minimize MSE. Only valid for
    continuous regression tasks.

    Returns
    -------
    probe : torch.nn.Linear
        Linear layer with weights set to the closed-form solution.
    final_train_metric : float
        Training MSE computed on the provided representations/labels.
    train_rmse : float
        Training RMSE computed on the provided representations/labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.tensor(representations, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

    N, D = x_tensor.shape
    x_with_bias = torch.cat([x_tensor, torch.ones(N, 1, device=device)], dim=1)  # (N, D+1)

    try:
        x_pinv = torch.linalg.pinv(x_with_bias)  # (D+1, N)
        optimal_weights = x_pinv @ y_tensor      # (D+1, 1)
        w = optimal_weights[:-1, 0]              # (D,)
        b = optimal_weights[-1, 0]               # scalar
    except Exception as e:
        print(f"Warning: Pseudoinverse failed ({e}), falling back to lstsq solution")
        solution = torch.linalg.lstsq(x_with_bias, y_tensor, rcond=None)
        optimal_weights = solution.solution
        w = optimal_weights[:-1, 0]
        b = optimal_weights[-1, 0]

    probe = torch.nn.Linear(D, 1).to(device)
    with torch.no_grad():
        probe.weight.copy_(w.unsqueeze(0))  # (1, D)
        probe.bias.copy_(b.unsqueeze(0))    # (1,)

    with torch.no_grad():
        preds = probe(x_tensor)
        mse_loss = torch.nn.functional.mse_loss(preds, y_tensor).item()
        mae_loss = torch.nn.functional.l1_loss(preds, y_tensor).item()
        rmse_loss = float(np.sqrt(mse_loss))

    print("Closed-form solution found:")
    print(f"  Training MSE: {mse_loss:.6f}")
    print(f"  Training RMSE: {rmse_loss:.6f}")
    print(f"  Training MAE: {mae_loss:.6f}")

    if args and hasattr(args, 'wandb') and args.wandb:
        wandb.log({
            "model/input_dim": D,
            "model/output_dim": 1,
            "dataset/num_samples": N,
            "train/method": "closed_form",
            "train/final_mse": mse_loss,
            "train/final_rmse": rmse_loss,
        })

    return probe, mse_loss, rmse_loss

def evaluate_probe(probe, test_representations, test_labels, args=None):
    """Evaluate the trained probe on a held-out set.

    Predictions are obtained by applying a sigmoid to the logits and using a
    0.5 threshold to map probabilities to the binary classes {0, 1}.
    
    Returns
    -------
    test_acc : float
        Test accuracy (binary classification accuracy).
    test_auc : float
        Test AUC score (Area Under the ROC Curve).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()
    
    # Check if test set is empty
    if len(test_representations) == 0:
        print("WARNING: Test dataset is empty! Cannot evaluate probe.")
        if args and args.wandb:
            wandb.log({
                "test/accuracy": 0.0,
                "test/auc": 0.0,
                "test/loss": 0.0,
                "test/precision": 0.0,
                "test/recall": 0.0,
                "test/f1": 0.0,
                "test/true_positives": 0,
                "test/false_positives": 0,
                "test/true_negatives": 0,
                "test/false_negatives": 0,
            })
        return 0.0, 0.0
    
    with torch.no_grad():
        test_representations = torch.tensor(test_representations, dtype=torch.float32, device=device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
        logits = probe(test_representations)
        probs = torch.sigmoid(logits)
        print(probs)
        preds = (probs > 0.5).float()
        test_acc = (preds == test_labels).float().mean().item()
        
        # Calculate AUC score using probabilities and true labels
        probs_cpu = probs.cpu().numpy().flatten()
        labels_cpu = test_labels.cpu().numpy().flatten()
        test_auc = roc_auc_score(labels_cpu, probs_cpu)
        
        # Calculate additional metrics
        test_loss = torch.nn.BCEWithLogitsLoss()(logits, test_labels).item()
        
        # Calculate precision, recall, and F1
        tp = ((preds == 1) & (test_labels == 1)).float().sum().item()
        fp = ((preds == 1) & (test_labels == 0)).float().sum().item()
        tn = ((preds == 0) & (test_labels == 0)).float().sum().item()
        fn = ((preds == 0) & (test_labels == 1)).float().sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Log test metrics to wandb
        if args and args.wandb:
            wandb.log({
                "test/accuracy": test_acc,
                "test/auc": test_auc,
                "test/loss": test_loss,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
                "test/true_positives": tp,
                "test/false_positives": fp,
                "test/true_negatives": tn,
                "test/false_negatives": fn,
            })
    
    return test_acc, test_auc


def evaluate_probe_continuous(probe, test_representations, test_labels, test_dms_datasets=None, args=None):
    """Evaluate the trained regression probe on a held-out set.
    
    Returns regression metrics: RMSE, MAE, R², Pearson correlation, and Spearman correlation.
    If test_dms_datasets is provided, also computes per-dataset Spearman correlations.
    
    Returns
    -------
    rmse : float
        Root Mean Squared Error
    mae : float  
        Mean Absolute Error
    r2 : float
        R-squared (coefficient of determination)
    pearson : float
        Pearson correlation coefficient
    spearman : float
        Overall Spearman rank correlation coefficient
    per_dataset_spearman : dict or None
        Per-dataset Spearman correlations (if test_dms_datasets provided)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.eval()
    
    # Check if test set is empty
    if len(test_representations) == 0:
        print("WARNING: Test dataset is empty! Cannot evaluate probe.")
        if args and args.wandb:
            wandb.log({
                "test/rmse": 0.0,
                "test/mae": 0.0,
                "test/r2": 0.0,
                "test/pearson": 0.0,
                "test/spearman": 0.0,
            })
        return 0.0, 0.0, 0.0, 0.0, 0.0, None
    
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
            
        # Overall Spearman correlation
        if np.std(y_true_np) > 0 and np.std(y_pred_np) > 0:
            try:
                spearman_result = spearmanr(y_true_np, y_pred_np)
                corr_attr = getattr(spearman_result, 'correlation', None)
                pval_attr = getattr(spearman_result, 'pvalue', None)
                if corr_attr is not None and pval_attr is not None:
                    spearman = float(corr_attr)
                    spearman_pvalue = float(pval_attr)
                else:
                    result_tuple = cast(Tuple[float, float], spearman_result)  # type: ignore[assignment]
                    spearman = float(result_tuple[0])
                    spearman_pvalue = float(result_tuple[1])
            except Exception as e:
                print(f"Warning: Failed to compute Spearman correlation: {e}")
                spearman = 0.0
                spearman_pvalue = 1.0
        else:
            spearman = 0.0
            spearman_pvalue = 1.0
        
        # Per-dataset Spearman correlations and average
        per_dataset_spearman = None
        avg_spearman = 0.0
        if test_dms_datasets is not None:
            per_dataset_spearman = {}
            unique_datasets = np.unique(test_dms_datasets)
            valid_spearmans = []
            
            print(f"\nPer-dataset Spearman correlations:")
            for dataset_id in unique_datasets:
                dataset_mask = test_dms_datasets == dataset_id
                dataset_y_true = y_true_np[dataset_mask]
                dataset_y_pred = y_pred_np[dataset_mask]
                
                if len(dataset_y_true) > 1 and np.std(dataset_y_true) > 0 and np.std(dataset_y_pred) > 0:
                    try:
                        dataset_spearman_result = spearmanr(dataset_y_true, dataset_y_pred)
                        corr_attr = getattr(dataset_spearman_result, 'correlation', None)
                        if corr_attr is not None:
                            dataset_spearman = float(corr_attr)
                        else:
                            result_tuple = cast(Tuple[float, float], dataset_spearman_result)  # type: ignore[assignment]
                            dataset_spearman = float(result_tuple[0])
                        per_dataset_spearman[dataset_id] = dataset_spearman
                        valid_spearmans.append(dataset_spearman)
                        print(f"  {dataset_id}: {dataset_spearman:.6f} (n={len(dataset_y_true)})")
                    except Exception as e:
                        print(f"  {dataset_id}: Failed to compute Spearman - {e}")
                        per_dataset_spearman[dataset_id] = np.nan
                else:
                    print(f"  {dataset_id}: Insufficient variance (n={len(dataset_y_true)})")
                    per_dataset_spearman[dataset_id] = np.nan

            # Calculate average Spearman from valid per-dataset results
            if valid_spearmans:
                avg_spearman = float(np.mean(valid_spearmans))
                print(f"\nAverage Spearman across {len(valid_spearmans)} datasets: {avg_spearman:.6f}")
            else:
                avg_spearman = 0.0
                print("\nNo valid per-dataset Spearman correlations computed")

            # Log per-dataset results to wandb
            if args and args.wandb:
                wandb.log({"test/avg_spearman": avg_spearman})
                for dataset_id, dataset_spearman in per_dataset_spearman.items():
                    if not np.isnan(dataset_spearman):
                        wandb.log({f"test/spearman_{dataset_id}": dataset_spearman})
        else:
            # Fallback to overall Spearman if no dataset identifiers
            avg_spearman = spearman
            print(f"\nNo dataset identifiers found, using overall Spearman: {avg_spearman:.6f}")

        # Log test metrics to wandb
        if args and args.wandb:
            wandb.log({
                "test/rmse": rmse,
                "test/mae": mae,
                "test/r2": r2,
                "test/pearson": pearson,
                "test/spearman_overall": spearman,  # Keep for reference
                "test/avg_spearman": avg_spearman,  # Main metric
                "test/spearman_pvalue": spearman_pvalue,
            })

    return rmse, mae, r2, pearson, avg_spearman, per_dataset_spearman


def save_probe_artifacts(probe, save_path, metadata):
    """Save a torch.nn.Linear probe and a metadata JSON next to it.

    - If save_path ends with .pt/.pth, treat it as a file path.
    - Otherwise, treat save_path as a directory and create a timestamped file.
    """
    save_path = Path(save_path)

    # Move to CPU for portable saving
    probe_cpu = probe.cpu()

    # Resolve output paths
    if save_path.suffix.lower() in {".pt", ".pth"}:
        weights_path = save_path
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = weights_path.with_suffix(".json")
    else:
        # Route directory saves into subfolders by mode/arch and model identifier
        mode = str(metadata.get("training_mode", "mode"))
        if mode == "closed_form":
            subfolder = "closed_form_val_spearman"  # Default closed form uses val_spearman
        elif mode == "closed_form_train_rmse":
            subfolder = "closed_form_train_rmse"
        elif bool(metadata.get("non_linear", False)):
            subfolder = "nonlinear"
        else:
            subfolder = "linear"
        # Create a per-model subfolder to avoid overwriting across different models
        model_id_raw = metadata.get("model_id")
        model_folder = None
        if model_id_raw is not None:
            try:
                model_folder = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_id_raw))
            except Exception:
                model_folder = str(model_id_raw)
        save_path = save_path / subfolder / (model_folder if model_folder else "default")
        save_path.mkdir(parents=True, exist_ok=True)
        task = str(metadata.get("task_type", "task"))
        mode = str(metadata.get("training_mode", "mode"))
        # Convert training mode to detailed token
        if mode == "epochs" and metadata.get("num_epochs") is not None:
            mode_token = f"epochs={int(metadata['num_epochs'])}"
        elif mode == "steps" and metadata.get("num_steps") is not None:
            mode_token = f"steps={int(metadata['num_steps'])}"
        else:
            mode_token = mode
        # Layer token
        layer_val = metadata.get("layer")
        layer_token = str(layer_val) if layer_val is not None else "x"
        # Loss token (force mse for closed_form modes)
        loss_name = str(metadata.get("loss_function", "mse"))
        if mode in ("closed_form", "closed_form_train_rmse"):
            loss_name = "mse"
        # Numerics formatting
        lr_val = metadata.get("learning_rate")
        try:
            lr_token = f"{float(lr_val):.10g}"
        except Exception:
            lr_token = str(lr_val)
        bs_token = str(metadata.get("batch_size", "x"))
        seed_token = str(metadata.get("seed", "x"))
        nonlinear_token = "_nonlinear" if bool(metadata.get("non_linear", False)) else ""
        stem = (
            f"dms_probe_{task}{nonlinear_token}_{mode_token}_layer={layer_token}"
            f"_loss={loss_name}_lr={lr_token}_batch_size={bs_token}_seed={seed_token}"
        )
        weights_path = save_path / f"{stem}.pt"
        metadata_path = save_path / f"{stem}.json"

    # Determine in/out features from the last Linear layer
    last_linear = None
    if isinstance(probe_cpu, torch.nn.Sequential):
        for module in reversed(probe_cpu):
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                break
    elif isinstance(probe_cpu, torch.nn.Linear):
        last_linear = probe_cpu
    if last_linear is None:
        raise RuntimeError("Expected probe to contain a torch.nn.Linear layer for output")

    in_features = int(last_linear.weight.shape[1])
    out_features = int(last_linear.weight.shape[0])

    torch.save({
        "state_dict": probe_cpu.state_dict(),
        "in_features": in_features,
        "out_features": out_features,
        "arch": "mlp" if isinstance(probe_cpu, torch.nn.Sequential) else "linear",
    }, weights_path)

    enriched = dict(metadata)
    enriched.update({
        "in_features": in_features,
        "out_features": out_features,
        "weights_path": str(weights_path),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "format_version": 1,
    })
    with metadata_path.open("w") as f:
        json.dump(enriched, f, indent=2, sort_keys=True)

    print(f"Saved probe weights to: {weights_path}")
    print(f"Saved probe metadata to: {metadata_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets/dms_probe_dataset_layer_26_train.h5")
    parser.add_argument("--test_dataset_path", type=str, default="/workspaces/BioRiskEval/attack/data/eval_dataset/virulence/probe_datasets/dms_probe_dataset_layer_26_test.h5")
    
    # Training duration options (mutually exclusive)
    training_group = parser.add_mutually_exclusive_group(required=False)
    training_group.add_argument("--num_steps", type=int, help="Number of training steps (default: 100)")
    training_group.add_argument("--num_epochs", type=int, help="Number of training epochs (alternative to --num_steps)")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # Task type and loss function arguments
    parser.add_argument("--task_type", type=str, default="binary", choices=["binary", "continuous"], 
                       help="Task type: binary classification or continuous regression")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "huber", "bce"], 
                       help="Loss function (for continuous: mse/mae/huber, for binary: bce)")
    parser.add_argument("--huber_delta", type=float, default=1.0, 
                       help="Delta parameter for Huber/SmoothL1 loss when --loss huber is selected")
    parser.add_argument("--closed_form", action="store_true",
                        help="Solve continuous regression in closed form (pseudoinverse), no training")
    parser.add_argument("--save_probe_path", type=str, default=None,
                        help="Path to save the probe (.pt/.pth or a directory). Saves a metadata JSON alongside.")
    # Model architecture options
    parser.add_argument("--non_linear", action="store_true", help="Use an MLP probe instead of a single linear layer")
    parser.add_argument("--hidden_dims", type=str, default="512", help="Comma-separated hidden layer sizes for MLP, e.g., '512,512'")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "tanh"], help="Activation for MLP hidden layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for MLP hidden layers")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--results_file", type=str, help="Path to save results CSV file")

    args = parser.parse_args()
    
    if not args.closed_form and args.num_steps is None and args.num_epochs is None:
        raise ValueError("Either --num_steps or --num_epochs must be specified (unless --closed_form)")
    
    # Initialize wandb
    if args.wandb:
        # Get project name from environment variable or use default
        wandb_project = os.getenv("WANDB_PROJECT")
        # Create config dict for wandb
        config = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
        # Add training mode to config
        if args.closed_form:
            config["training_mode"] = "closed_form"
        elif args.num_epochs is not None:
            config["num_epochs"] = args.num_epochs
            config["training_mode"] = "epochs"
        else:
            config["num_steps"] = args.num_steps
            config["training_mode"] = "steps"
        
        wandb.init(
            project=wandb_project,
            config=config,
        )
        if wandb.run is not None:
            print(f"🚀 Wandb run initialized: {wandb.run.name}")
        else:
            print("🚀 Wandb run initialized")
    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_sequences, train_representations, train_labels, train_dms_datasets = read_probe_dataset(args.train_dataset_path)
    test_sequences, test_representations, test_labels, test_dms_datasets = read_probe_dataset(args.test_dataset_path)

    if args.closed_form:
        if args.task_type != 'continuous':
            raise ValueError("--closed_form is only supported with --task_type continuous")
        if getattr(args, 'non_linear', False):
            raise ValueError("--closed_form is not compatible with --non_linear (closed-form only supports linear regression)")
        if hasattr(args, 'loss') and args.loss != 'mse':
            print("Warning: --closed_form ignores --loss and uses MSE.")
        probe, final_train_metric, train_rmse = solve_closed_form_probe(train_representations, train_labels, args)
    else:
        probe, final_train_metric = train_linear_probe(train_representations, train_labels, args)
        train_rmse = None  # Not calculated for iterative training
    
    # Route to appropriate evaluation function based on task type
    task_type = getattr(args, 'task_type', 'binary')
    
    probe_metrics = {}
    if task_type == 'binary':
        test_acc, test_auc = evaluate_probe(probe, test_representations, test_labels, args)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Final training accuracy: {final_train_metric:.4f}")
        probe_metrics = {
            "final_train_accuracy": float(final_train_metric),
            "test_accuracy": float(test_acc),
            "test_auc": float(test_auc),
        }
    elif task_type == 'continuous':
        test_rmse, test_mae, test_r2, test_pearson, avg_spearman, per_dataset_spearman = evaluate_probe_continuous(
            probe, test_representations, test_labels, test_dms_datasets, args)
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test R²: {test_r2:.6f}")
        print(f"Test Pearson: {test_pearson:.6f}")
        print(f"Average Spearman: {avg_spearman:.6f}")
        print(f"Final training loss: {final_train_metric:.6f}")
        per_dataset_clean = None
        if per_dataset_spearman is not None:
            per_dataset_clean = {}
            for k, v in per_dataset_spearman.items():
                key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                if not np.isnan(v):
                    per_dataset_clean[key_str] = float(v)
        probe_metrics = {
            "final_train_loss": float(final_train_metric),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "test_pearson": float(test_pearson),
            "avg_spearman": float(avg_spearman),
            "per_dataset_spearman": per_dataset_clean,
        }

    # Optionally save the probe and metadata
    if args.save_probe_path is not None:
        training_mode = "closed_form" if args.closed_form else ("epochs" if args.num_epochs is not None else "steps")
        wandb_run_name = "no_wandb"
        if args.wandb and wandb.run is not None:
            wandb_run_name = wandb.run.name
        # Try to infer layer from dataset filename(s)
        layer_inferred = None
        try:
            layer_pattern = re.compile(r"layer[_=]?(\d+)")
            for p in (args.train_dataset_path, args.test_dataset_path):
                m = layer_pattern.search(Path(p).name)
                if m:
                    layer_inferred = int(m.group(1))
                    break
        except Exception:
            layer_inferred = None
        metadata = {
            "task_type": task_type,
            "training_mode": training_mode,
            "loss_function": getattr(args, 'loss', 'mse') if task_type == 'continuous' else 'bce',
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "num_steps": int(args.num_steps) if args.num_steps is not None else None,
            "num_epochs": int(args.num_epochs) if args.num_epochs is not None else None,
            "layer": layer_inferred,
            "train_dataset": Path(args.train_dataset_path).name,
            "test_dataset": Path(args.test_dataset_path).name,
            # add model identifier derived from dataset folder to avoid overwrite
            "model_id": Path(args.train_dataset_path).parent.name if args.train_dataset_path else None,
            "wandb_run": wandb_run_name,
            "metrics": probe_metrics,
            "non_linear": bool(getattr(args, 'non_linear', False)),
            "hidden_dims": str(getattr(args, 'hidden_dims', '512')) if getattr(args, 'non_linear', False) else None,
            "activation": str(getattr(args, 'activation', 'relu')) if getattr(args, 'non_linear', False) else None,
            "dropout": float(getattr(args, 'dropout', 0.0)) if getattr(args, 'non_linear', False) else None,
        }
        save_probe_artifacts(probe, args.save_probe_path, metadata)
        
        # Print per-dataset summary if available
        if per_dataset_spearman is not None:
            valid_spearmans = [sp for sp in per_dataset_spearman.values() if not np.isnan(sp)]
            if valid_spearmans:
                print(f"Per-dataset summary: {len(valid_spearmans)} valid datasets")
                print(f"Spearman std across datasets: {np.std(valid_spearmans):.6f}")
    # Finish wandb run
    if args.wandb:
        wandb.finish()

    # Resolve results CSV location with backward-compatible routing
    # - If results_file ends with .csv, use it as-is (backward compatible)
    # - Otherwise, treat it as a directory (or default base) and write under
    #   subfolders: binary (linear), continuous (linear), nonlinear (continuous), closed_form
    base_results = Path(args.results_file) if args.results_file is not None else Path("ft-attack/probe_results")
    if base_results.suffix.lower() == ".csv":
        result_file = base_results
    else:
        # Determine subfolder by mode/arch with backward compatibility
        if task_type == 'binary':
            subfolder = 'binary'
        else:
            if args.closed_form:
                subfolder = 'closed_form'
            elif bool(getattr(args, 'non_linear', False)):
                subfolder = 'nonlinear'
            else:
                subfolder = 'continuous'
        result_dir = base_results / subfolder
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "probe_results.csv"
    
    # Append results to CSV, create file if it does not exist
    file_exists = result_file.exists()
    
    # Create directory if it doesn't exist
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get wandb run name if wandb is enabled
    wandb_run_name = "no_wandb"
    if args.wandb and wandb.run is not None:
        wandb_run_name = wandb.run.name
    
    with result_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Create headers based on task type
            if task_type == 'binary':
                # Preserve original binary schema
                writer.writerow([
                    "train_dataset_path", 
                    "test_dataset_path",
                    "learning_rate", 
                    "batch_size", 
                    "num_steps",
                    "num_epochs",
                    "task_type",
                    "final_train_metric",
                    "test_acc",
                    "test_auc",
                    "wandb_run",
                ])
            elif task_type == 'continuous':
                if args.closed_form:
                    # Closed form schema with train_rmse
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
                elif bool(getattr(args, 'non_linear', False)):
                    # Extended schema for nonlinear probes (append extra columns)
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
                        "test_rmse",
                        "test_mae", 
                        "test_r2",
                        "test_pearson",
                        "avg_spearman",
                        "per_dataset_spearman",
                        "wandb_run",
                        "non_linear",
                        "hidden_dims",
                        "activation",
                        "dropout",
                    ])
                else:
                    # Preserve original continuous schema
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
                        "test_rmse",
                        "test_mae", 
                        "test_r2",
                        "test_pearson",
                        "avg_spearman",
                        "per_dataset_spearman",
                        "wandb_run",
                    ])
        
        # Write data row based on task type
        # Nonlinear CSV extras
        csv_non_linear = bool(getattr(args, 'non_linear', False))
        csv_hidden_dims = str(getattr(args, 'hidden_dims', '512')) if csv_non_linear else ""
        csv_activation = str(getattr(args, 'activation', 'relu')) if csv_non_linear else ""
        csv_dropout = float(getattr(args, 'dropout', 0.0)) if csv_non_linear else ""
        if task_type == 'binary':
            writer.writerow([
                args.train_dataset_path.split("/")[-1].replace(".h5", ""),
                args.test_dataset_path.split("/")[-1].replace(".h5", ""),
                args.learning_rate,
                args.batch_size,
                args.num_steps if args.num_steps is not None else "None",
                args.num_epochs if args.num_epochs is not None else "None",
                task_type,
                f"{final_train_metric:.4f}",
                f"{test_acc:.4f}",
                f"{test_auc:.4f}",
                wandb_run_name,
            ])
        elif task_type == 'continuous':
            # Prepare per-dataset Spearman as JSON string
            per_dataset_json = "{}"
            if per_dataset_spearman is not None:
                # Filter out NaN values and convert bytes keys to strings for JSON serialization
                valid_per_dataset = {}
                for k, v in per_dataset_spearman.items():
                    if not np.isnan(v):
                        # Convert bytes keys to strings if needed
                        key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                        valid_per_dataset[key_str] = v
                per_dataset_json = json.dumps(valid_per_dataset, separators=(',', ':'))
            
            if args.closed_form:
                # Closed form data row with train_rmse
                base_row = [
                    args.train_dataset_path.split("/")[-1].replace(".h5", ""),
                    args.test_dataset_path.split("/")[-1].replace(".h5", ""),
                    args.learning_rate,
                    args.batch_size,
                    args.num_steps if args.num_steps is not None else "None",
                    args.num_epochs if args.num_epochs is not None else "None",
                    task_type,
                    getattr(args, 'loss', 'mse'),
                    f"{final_train_metric:.6f}",
                    f"{train_rmse:.6f}",
                    f"{test_rmse:.6f}",
                    f"{test_mae:.6f}",
                    f"{test_r2:.6f}",
                    f"{test_pearson:.6f}",
                    f"{avg_spearman:.6f}",
                    per_dataset_json,
                    wandb_run_name,
                ]
            else:
                # Regular continuous data row
                base_row = [
                    args.train_dataset_path.split("/")[-1].replace(".h5", ""),
                    args.test_dataset_path.split("/")[-1].replace(".h5", ""),
                    args.learning_rate,
                    args.batch_size,
                    args.num_steps if args.num_steps is not None else "None",
                    args.num_epochs if args.num_epochs is not None else "None",
                    task_type,
                    getattr(args, 'loss', 'mse'),
                    f"{final_train_metric:.6f}",
                    f"{test_rmse:.6f}",
                    f"{test_mae:.6f}",
                    f"{test_r2:.6f}",
                    f"{test_pearson:.6f}",
                    f"{avg_spearman:.6f}",
                    per_dataset_json,
                    wandb_run_name,
                ]
            if csv_non_linear:
                base_row.extend([
                    str(csv_non_linear),
                    csv_hidden_dims,
                    csv_activation,
                    csv_dropout,
                ])
            writer.writerow(base_row)    