import os
import argparse
import pandas as pd
import numpy as np
import torch
import sys
import time
import re
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score
from pathlib import Path
import gc
import h5py
from typing import Tuple, List
import json
import tempfile
import glob
import traceback

from nemo import lightning as nl

from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import InMemoryProteinDataset
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.utils.callbacks import PredictionWriter


def stratified_sample_dms(df, n_samples, stratify_column='DMS_score_bin', random_state=42):
    if len(df) <= n_samples:
        return df
    if stratify_column not in df.columns:
        return df.sample(n=n_samples, random_state=random_state)
    df_clean = df[df[stratify_column].notna()].copy()
    if len(df_clean) == 0:
        return df.sample(n=n_samples, random_state=random_state)
    value_counts = df_clean[stratify_column].value_counts()
    proportions = value_counts / len(df_clean)
    target_samples = {}
    total_allocated = 0
    for value, proportion in proportions.items():
        target = int(np.round(proportion * n_samples))
        target_samples[value] = max(1, target)
        total_allocated += target_samples[value]
    if total_allocated != n_samples:
        largest_group = max(target_samples.keys(), key=lambda x: target_samples[x])
        target_samples[largest_group] += n_samples - total_allocated
        target_samples[largest_group] = max(1, target_samples[largest_group])
    sampled_dfs = []
    for value, n_group_samples in target_samples.items():
        group_df = df_clean[df_clean[stratify_column] == value]
        if len(group_df) < n_group_samples:
            sampled_dfs.append(group_df)
        else:
            sampled_group = group_df.sample(n=n_group_samples, random_state=random_state)
            sampled_dfs.append(sampled_group)
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    return result_df


def get_model_name(args):
    if "ft_checkpoints" in str(args.ckpt_dir):
        training_config = args.ckpt_dir.parent.parent.parent.name
        checkpoint_name = args.ckpt_dir.stem
        samples_match = re.search(r'consumed_samples=(\d+)', checkpoint_name)
        if samples_match:
            samples_value = samples_match.group(1)
            model_name = f"{training_config}_samples={samples_value}"
        else:
            model_name = f"{training_config}_{checkpoint_name}"
    else:
        model_name = args.ckpt_dir.stem
    return model_name


def get_experiment_type(args):
    if hasattr(args, 'DMS_filenames') and args.DMS_filenames:
        filename = os.path.basename(args.DMS_filenames)
        experiment_type = os.path.splitext(filename)[0]
        return experiment_type
    elif hasattr(args, 'DMS_h5') and args.DMS_h5:
        return "virus_reproduction"
    else:
        return "single_file"


def get_likelihood_results_path(args):
    model_name = get_model_name(args)
    file_name = args.DMS_id + "_" + model_name + "_likelihoods.csv"
    experiment_type = get_experiment_type(args)
    if hasattr(args, 'down_sample') and args.down_sample is not None:
        sample_folder = f"sample={args.down_sample}_seed=42"
    else:
        sample_folder = "full"
    results_dir = os.path.join(
        args.output_performance_file_folder,
        experiment_type,
        sample_folder,
        args.taxon,
        model_name,
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, file_name)
    return output_path


def get_fitness_results_path(args):
    model_name = get_model_name(args)
    file_name = args.DMS_id + "_" + model_name + "_fitness.csv"
    if hasattr(args, 'h5_path') and args.h5_path:
        n_samples, seed, split = get_h5_metadata(args.h5_path)
        sample_folder = f"h5_samples={n_samples}_seed={seed}_{split}"
    else:
        if hasattr(args, 'down_sample') and args.down_sample is not None:
            sample_folder = f"sample={args.down_sample}_seed=42"
        else:
            sample_folder = "full"
    experiment_type = get_experiment_type(args)
    results_dir = os.path.join(
        args.output_performance_file_folder,
        experiment_type,
        sample_folder,
        args.taxon,
        model_name,
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, file_name)
    return output_path


def get_performance_results(merged_df, DMS_score_column, model_score_column, DMS_binary_score_column):
    performance_results = {}
    required_cols = [DMS_score_column, model_score_column]
    required_cols = [col for col in required_cols if col not in merged_df.columns]
    if required_cols:
        raise ValueError(f"Missing required columns: {required_cols}")
    y_true = merged_df[DMS_score_column]
    y_pred = merged_df[model_score_column]
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    if valid_mask.sum() == 0:
        return {key: np.nan for key in ['spearman', 'spearman_pvalue', 'ndcg', 'auc', 'mcc']}
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    try:
        spearman_result = spearmanr(y_true_valid, y_pred_valid)
        performance_results['spearman'] = spearman_result[0]
        performance_results['spearman_pvalue'] = spearman_result[1]
    except Exception:
        performance_results['spearman'] = np.nan
        performance_results['spearman_pvalue'] = np.nan
    try:
        performance_results['ndcg'] = ndcg_score(np.asarray([y_true_valid]), np.asarray([y_pred_valid]))
    except Exception:
        performance_results['ndcg'] = np.nan
    if DMS_binary_score_column in merged_df.columns:
        merged_df_no_nan_binary = merged_df[merged_df[DMS_binary_score_column].notna()].copy()
        if (len(merged_df_no_nan_binary) > 0 and len(merged_df_no_nan_binary[DMS_binary_score_column].unique()) > 1):
            try:
                y_true_binary = merged_df_no_nan_binary[DMS_binary_score_column]
                y_pred_binary = merged_df_no_nan_binary[model_score_column]
                performance_results['auc'] = roc_auc_score(y_true_binary, y_pred_binary)
                y_pred_binary_thresh = (y_pred_binary > np.median(y_pred_binary)).astype(int)
                performance_results['mcc'] = matthews_corrcoef(y_true_binary, y_pred_binary_thresh)
            except Exception:
                performance_results['auc'] = np.nan
                performance_results['mcc'] = np.nan
        else:
            performance_results['auc'] = np.nan
            performance_results['mcc'] = np.nan
    else:
        performance_results['auc'] = np.nan
        performance_results['mcc'] = np.nan
    return performance_results


def read_dms_filenames_from_csv(csv_path, column_name):
    try:
        DMS_filenames_df = pd.read_csv(csv_path)
        if column_name not in DMS_filenames_df.columns:
            raise ValueError(f"Column '{column_name}' not found in {csv_path}")
        DMS_filenames = DMS_filenames_df[column_name].tolist()
        return DMS_filenames
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")


def read_dms_filenames_from_text(text_path):
    try:
        with open(text_path, 'r') as f:
            DMS_filenames = f.read().splitlines()
        return DMS_filenames
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading DMS filenames from {text_path}: {e}")


def read_dms_data_from_h5(h5_path: str) -> pd.DataFrame:
    required_cols = ['sequences', 'scores', 'DMS_dataset', 'labels']
    try:
        with h5py.File(h5_path, 'r') as f:
            missing_datasets = [ds for ds in required_cols if ds not in f]
            if missing_datasets:
                raise KeyError(f"Missing required datasets in HDF5 file: {missing_datasets}")
            dms_datasets = np.array(f['DMS_dataset'])
            sequences = np.array(f['sequences'])
            scores = np.array(f['scores'])
            labels = np.array(f['labels'])
            dms_datasets = np.char.decode(dms_datasets.astype('S'), 'utf-8')
            sequences = np.char.decode(sequences.astype('S'), 'utf-8')
        df = pd.DataFrame({
            'DMS_dataset': dms_datasets,
            'mutated_sequence': sequences,
            'DMS_score': scores,
            'DMS_score_bin': labels,
        })
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file {h5_path}: {e}")


def get_h5_metadata(h5_path: str) -> Tuple[int, int, str]:
    json_path = h5_path.replace('.h5', '.json').replace('dms_probe_dataset_', 'dataset_metadata_')
    with open(json_path, 'r') as f:
        metadata = json.loads(f.read())
        return (
            metadata['n_samples_per_dataset'],
            metadata['seed'],
            metadata['split'],
        )


def get_taxon_from_reference(DMS_id: str, DMS_reference_df: pd.DataFrame) -> str:
    try:
        taxon_mask = DMS_reference_df['DMS_id'] == DMS_id
        taxon_series = DMS_reference_df[taxon_mask]['taxon']
        raw_taxon = taxon_series.tolist()[0]
        return raw_taxon.capitalize()
    except Exception:
        return "Unknown"


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()


def get_wildtype_sequence_from_reference(DMS_id: str, DMS_reference_df: pd.DataFrame) -> str | None:
    """Lookup WT protein sequence using reference file.

    Prefers matching on `DMS_filename` (i.e., `${DMS_id}.csv`), falls back to `DMS_id`.
    """
    try:
        # Try match by filename first
        fname = f"{DMS_id}.csv"
        by_filename = DMS_reference_df[DMS_reference_df['DMS_filename'] == fname]
        if len(by_filename) > 0 and 'target_seq' in by_filename.columns:
            return str(by_filename.iloc[0]['target_seq'])
        # Fallback: match by DMS_id
        by_id = DMS_reference_df[DMS_reference_df['DMS_id'] == DMS_id]
        if len(by_id) > 0 and 'target_seq' in by_id.columns:
            return str(by_id.iloc[0]['target_seq'])
    except Exception:
        pass
    return None


def apply_mutations_to_sequence(mutation_spec: str, wt_sequence: str, offset_idx: int = 1) -> str:
    """Apply colon-separated mutations like 'A10V' to a wildtype sequence.

    Args:
        mutation_spec: Mutation string, possibly with ':' separators for multiple mutations.
        wt_sequence: Wildtype amino acid sequence.
        offset_idx: 1 for 1-based indexing (ProteinGym), 0 for 0-based.
    """
    seq_chars = list(wt_sequence)
    for mut in str(mutation_spec).split(":"):
        if not mut:
            continue
        wt, mt = mut[0], mut[-1]
        pos = int(mut[1:-1]) - offset_idx
        if pos < 0 or pos >= len(seq_chars):
            raise ValueError(f"Mutation position out of bounds: {mut} on length {len(seq_chars)}")
        if seq_chars[pos] != wt:
            raise ValueError(f"WT mismatch at {pos}: expected {wt}, found {seq_chars[pos]}")
        seq_chars[pos] = mt
    return "".join(seq_chars)


def get_mut_offset_from_reference(DMS_id: str, DMS_reference_df: pd.DataFrame, default_offset: int = 1) -> int:
    """Fetch mutation index offset (1-based default) from reference file if available."""
    try:
        fname = f"{DMS_id}.csv"
        by_filename = DMS_reference_df[DMS_reference_df['DMS_filename'] == fname]
        if len(by_filename) > 0 and 'raw_mut_offset' in by_filename.columns:
            raw = by_filename.iloc[0]['raw_mut_offset']
            if pd.notna(raw):
                return int(raw)
        by_id = DMS_reference_df[DMS_reference_df['DMS_id'] == DMS_id]
        if len(by_id) > 0 and 'raw_mut_offset' in by_id.columns:
            raw = by_id.iloc[0]['raw_mut_offset']
            if pd.notna(raw):
                return int(raw)
    except Exception:
        pass
    return default_offset


def _load_esm2_predictions(results_dir: str) -> dict:
    epoch_files = sorted(glob.glob(os.path.join(results_dir, 'predictions__rank_*.pt')))
    batch_files = sorted(glob.glob(os.path.join(results_dir, 'predictions__rank_*__batch_*.pt')))
    if epoch_files:
        parts = [torch.load(f, map_location='cpu') for f in epoch_files]
        # Concatenate along batch dimension for each key
        merged = {}
        for key in parts[0].keys():
            if parts[0][key] is None:
                merged[key] = None
            else:
                merged[key] = torch.cat([p[key] for p in parts], dim=0)
        return merged
    elif batch_files:
        # Collate batch shards per rank
        by_rank = {}
        for f in batch_files:
            m = re.search(r'predictions__rank_(\d+)__batch_\d+\.pt$', f)
            if not m:
                continue
            r = int(m.group(1))
            by_rank.setdefault(r, []).append(f)
        parts = []
        for r in sorted(by_rank.keys()):
            rank_parts = [torch.load(f, map_location='cpu') for f in sorted(by_rank[r])]
            # For batch mode, ordering may not be consistent; we still concatenate
            merged_rank = {}
            for key in rank_parts[0].keys():
                if rank_parts[0][key] is None:
                    merged_rank[key] = None
                else:
                    merged_rank[key] = torch.cat([p[key] for p in rank_parts], dim=0)
            parts.append(merged_rank)
        merged = {}
        for key in parts[0].keys():
            if parts[0][key] is None:
                merged[key] = None
            else:
                merged[key] = torch.cat([p[key] for p in parts], dim=0)
        return merged
    else:
        raise FileNotFoundError(f"No predictions found in {results_dir}")


def _compute_esm2_log_likelihoods_from_predictions(predictions: dict, tokenizer, reduce_method: str) -> List[float]:
    token_logits = predictions.get('token_logits')
    input_ids = predictions.get('input_ids')
    if token_logits is None or input_ids is None:
        raise RuntimeError("Predictions must include 'token_logits' and 'input_ids'")
    # token_logits: [seq_len, batch, vocab] -> [batch, seq_len, vocab]
    logits = token_logits.transpose(0, 1).contiguous()
    log_probs = torch.log_softmax(logits, dim=-1)
    # Gather log probs of the true token at each position
    gathered = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    # Mask special tokens (e.g., CLS/EOS/PAD)
    special_ids = torch.tensor(tokenizer.all_special_ids, dtype=input_ids.dtype)
    mask = ~torch.isin(input_ids, special_ids)
    gathered = gathered.masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=1).clamp(min=1)
    if reduce_method == 'mean':
        seq_scores = gathered.sum(dim=1) / lengths
    else:
        seq_scores = gathered.sum(dim=1)
    return seq_scores.cpu().tolist()


_MEGATRON_INIT_DONE = False
_RUNTIME = None  # cached (tokenizer, module, trainer, global_batch_size)


def _get_inference_device(module) -> torch.device:
    """Pick the actual device of the instantiated inner model if available; fallback to CUDA/CPU.

    Handles Megatron Lightning wrapper where parameters may only exist after configure_model().
    """
    try:
        inner = getattr(module, 'module', None)
        if inner is not None:
            # Try parameter device
            try:
                first_param = next(inner.parameters())
                return first_param.device
            except StopIteration:
                pass
            # Try buffer device
            try:
                for _, buf in inner.named_buffers():
                    return buf.device
            except Exception:
                pass
    except Exception:
        pass
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_distributed_and_model_parallel(args) -> None:
    global _MEGATRON_INIT_DONE
    if _MEGATRON_INIT_DONE:
        return
    try:
        import torch.distributed as dist
        if dist.is_available() and not dist.is_initialized():
            # Ensure required env vars for env:// init
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)
        try:
            from megatron.core import parallel_state as mps
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
            # Initialize model parallel (no-op if already initialized)
            try:
                initialized = getattr(mps, 'model_parallel_is_initialized', None)
                if initialized is None or not initialized():
                    mps.initialize_model_parallel(
                        tensor_model_parallel_size=getattr(args, 'tensor_parallel_size', 1),
                        pipeline_model_parallel_size=getattr(args, 'pipeline_model_parallel_size', 1),
                    )
            except Exception:
                # Fallback if API differs
                mps.initialize_model_parallel(
                    tensor_model_parallel_size=getattr(args, 'tensor_parallel_size', 1),
                    pipeline_model_parallel_size=getattr(args, 'pipeline_model_parallel_size', 1),
                )
            try:
                # Seed Megatron RNGs so CUDA RNG tracker has expected states
                base_seed = int(torch.initial_seed() % (2**31 - 1))
                model_parallel_cuda_manual_seed(base_seed)
            except Exception as e:
                print(f"[debug] Warning: Could not seed Megatron RNG: {type(e).__name__}: {e}")
            try:
                # Sync once to ensure ranks agree, even in single-process setup
                if dist.is_initialized():
                    dist.barrier()
            except Exception:
                pass
        except Exception as e:
            print(f"[debug] Warning: Could not initialize Megatron model-parallel state: {type(e).__name__}: {e}")
        _MEGATRON_INIT_DONE = True
    except Exception as e:
        print(f"[debug] Warning: Could not initialize distributed process group: {type(e).__name__}: {e}")


def _init_runtime_once(args):
    """Initialize tokenizer, module, trainer once, reusing them across files."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    tokenizer = get_tokenizer()
    # Trainer-style config (ensure logits + input_ids enabled)
    config = ESM2Config(
        include_hiddens=False,
        include_embeddings=False,
        include_input_ids=True,
        skip_logits=False,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        initial_ckpt_path=str(args.ckpt_dir),
        params_dtype=torch.bfloat16 if args.precision == "bf16-mixed" else torch.float32,
        pipeline_dtype=torch.bfloat16 if args.precision == "bf16-mixed" else torch.float32,
        autocast_dtype=torch.bfloat16 if args.precision == "bf16-mixed" else torch.float32,
    )
    module = biobert_lightning_module(config=config, tokenizer=tokenizer)
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_parallel_load=True,
    )
    trainer = nl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=strategy,
        num_nodes=args.num_nodes,
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
        max_steps=1,
    )
    global_batch_size = infer_global_batch_size(
        micro_batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        devices=args.devices,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    )
    _RUNTIME = {
        "tokenizer": tokenizer,
        "module": module,
        "trainer": trainer,
        "global_batch_size": global_batch_size,
    }
    return _RUNTIME


def _init_esm2_module(args):
    try:
        tokenizer = get_tokenizer()
        config = ESM2Config(
            include_hiddens=False,
            include_embeddings=False,
            include_input_ids=False,
            skip_logits=False,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            initial_ckpt_path=str(args.ckpt_dir),
        )
        print(f"[debug] Initializing ESM2 module from '{str(args.ckpt_dir)}' (TP={args.tensor_parallel_size}, PP={args.pipeline_model_parallel_size})")
        module = biobert_lightning_module(config=config, tokenizer=tokenizer)
        # Ensure distributed and model-parallel states are initialized for Megatron-backed module
        _ensure_distributed_and_model_parallel(args)
        # Force instantiation of underlying Megatron model before moving to device
        try:
            module.configure_model()
        except Exception:
            # configure_model is idempotent and called again in forward; best-effort here
            pass
        module.eval()
        device = _get_inference_device(module)
        try:
            # Move instantiated inner module if present
            if hasattr(module, 'module') and module.module is not None:
                module.module.to(device)
        except Exception:
            pass
        try:
            module.to(device)
        except Exception:
            pass
        print(f"[debug] Inference device set to: {device}")
        return module, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ESM2 module from '{str(args.ckpt_dir)}': {type(e).__name__}: {e}")


@torch.no_grad()
def _score_pseudo_ppl_batch(seqs: List[str], module, tokenizer) -> List[float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = []
    for seq in seqs:
        input_ids = tokenizer.encode(seq, add_special_tokens=True, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.flatten().to(device)
        seq_len = input_ids.size(0)
        # positions to mask: exclude specials at 0 and last index
        if seq_len <= 2:
            scores.append(float("nan"))
            continue
        positions = torch.arange(1, seq_len - 1, device=device)
        batch = input_ids.unsqueeze(0).repeat(len(positions), 1)
        batch[torch.arange(len(positions)), positions] = tokenizer.mask_token_id
        attention_mask = torch.ones_like(batch, device=device)
        out = module(batch, attention_mask)
        logits = out["token_logits"].transpose(0, 1).contiguous()  # [B, S, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        true_tokens = input_ids[positions]
        gathered = log_probs[torch.arange(len(positions)), positions, true_tokens]
        scores.append(gathered.sum().item())
    return scores


@torch.no_grad()
def _score_masked_marginals(wt_seq: str, module, tokenizer) -> torch.Tensor:
    """Return tensor of shape [seq_len_tokenized, vocab] with log-probs at each position."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer.encode(wt_seq, add_special_tokens=True, return_tensors="pt")
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.flatten().to(device)
    seq_len = input_ids.size(0)
    positions = torch.arange(0, seq_len, device=device)
    # Only mask non-special positions
    special = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids, device=device))
    mask_positions = positions[~special]
    batch = input_ids.unsqueeze(0).repeat(len(mask_positions), 1)
    batch[torch.arange(len(mask_positions)), mask_positions] = tokenizer.mask_token_id
    attention_mask = torch.ones_like(batch, device=device)
    out = module(batch, attention_mask)
    if not isinstance(out, dict) or "token_logits" not in out:
        raise KeyError(f"Model output missing 'token_logits'. Keys present: {list(out.keys()) if isinstance(out, dict) else type(out)}")
    logits = out["token_logits"].transpose(0, 1).contiguous()  # [B, S, V] after transpose -> [S, B, V]?
    # After transpose above: original was [S, B, V]; we want [B, S, V]
    # Recompute to [B, S, V]
    logits = out["token_logits"].transpose(0, 1).contiguous()
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, S, V]
    # Gather row per masked position at that same position index
    mm = torch.empty((seq_len, log_probs.size(-1)), device=device).fill_(float("-inf"))
    for i, pos in enumerate(mask_positions.tolist()):
        mm[pos] = log_probs[i, pos]
    return mm  # [S, V]


def _score_masked_marginals_with_trainer(wt_seq: str, args) -> dict[int, torch.Tensor]:
    """Use Lightning Trainer+MegatronStrategy (as in infer_esm2.py) to get log-probs at masked positions.

    Returns a mapping: {aa_index -> log_prob_vector(torch.Tensor[vocab])}.
    """
    # Build masked sequences, one per amino-acid position
    mask_token = get_tokenizer().mask_token
    masked_sequences: List[str] = []
    for i, ch in enumerate(wt_seq):
        seq_chars = list(wt_seq)
        seq_chars[i] = mask_token
        masked_sequences.append("".join(seq_chars))

    # Write to temp CSV for dataset.from_csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df_tmp = pd.DataFrame({"sequences": masked_sequences})
        df_tmp.to_csv(tmp.name, index=False)
        temp_csv = tmp.name

    try:
        rt = _init_runtime_once(args)
        tokenizer = rt["tokenizer"]
        # Dataset/Datamodule for this sequence only
        dataset = InMemoryProteinDataset.from_csv(temp_csv, ignore_labels=True)
        datamodule = ESM2FineTuneDataModule(
            predict_dataset=dataset,
            micro_batch_size=args.batch_size,
            global_batch_size=rt["global_batch_size"],
            min_seq_length=args.min_seq_length,
            max_seq_length=args.min_seq_length,
            tokenizer=tokenizer,
        )
        preds = rt["trainer"].predict(rt["module"], datamodule=datamodule)

        # Collate predictions
        # preds is a list over batches; each item likely a dict with token_logits [S, B, V] and input_ids [B, S]
        batch_lp = []
        for out in preds:
            if isinstance(out, list):
                for o in out:
                    batch_lp.append(o)
            else:
                batch_lp.append(out)

        token_logits_list = []
        input_ids_list = []
        for o in batch_lp:
            if o is None:
                continue
            tl = o.get("token_logits") if isinstance(o, dict) else None
            ii = o.get("input_ids") if isinstance(o, dict) else None
            if tl is None or ii is None:
                continue
            # Convert to [B, S, V]
            token_logits_list.append(tl.transpose(0, 1).contiguous())
            input_ids_list.append(ii)

        if not token_logits_list:
            raise RuntimeError("No predictions with token_logits/input_ids returned from trainer.predict")

        token_logits = torch.cat(token_logits_list, dim=0)  # [N, S, V]
        input_ids = torch.cat(input_ids_list, dim=0)        # [N, S]

        # Map each sample (position i) to log-prob vector at mask token position
        pos_to_lp: dict[int, torch.Tensor] = {}
        mask_id = tokenizer.mask_token_id
        for idx_sample in range(token_logits.size(0)):
            ids_row = input_ids[idx_sample]
            mask_positions = (ids_row == mask_id).nonzero(as_tuple=False).flatten()
            if mask_positions.numel() != 1:
                continue
            pos = int(mask_positions.item())
            lp_vec = torch.log_softmax(token_logits[idx_sample, pos], dim=-1)
            pos_to_lp[idx_sample] = lp_vec.cpu()
        return pos_to_lp
    finally:
        try:
            os.remove(temp_csv)
        except Exception:
            pass
        # Aggressive cleanup of per-call objects
        try:
            del dataset
            del datamodule
        except Exception:
            pass
        cleanup_memory()

def get_esm2_likelihoods(DMS_df: pd.DataFrame, args, mutation_col: str = "mutated_sequence") -> pd.DataFrame:
    print(f"[debug] Scoring strategy: {args.scoring_strategy}")
    print(f"[debug] DMS_df.shape={DMS_df.shape}; columns={list(DMS_df.columns)}")
    module, tokenizer = _init_esm2_module(args)
    if args.scoring_strategy == "pseudo-ppl":
        if mutation_col not in DMS_df.columns:
            raise ValueError(f"DMS dataframe must contain '{mutation_col}' column for pseudo-ppl evaluation")
        sequences = DMS_df[mutation_col].astype(str).tolist()
        scores = _score_pseudo_ppl_batch(sequences, module, tokenizer)
        result_df = pd.DataFrame({"log_likelihood": scores}, index=DMS_df.index)
    elif args.scoring_strategy == "masked-marginals":
        # Need WT sequence and mutant column
        if args.mutant_col not in DMS_df.columns:
            raise ValueError(f"DMS dataframe must contain mutant column '{args.mutant_col}' for masked-marginals")
        wt_seq = get_wildtype_sequence_from_reference(args.DMS_id, args.DMS_reference_df)
        if wt_seq is None:
            raise ValueError(f"Could not find target_seq for DMS_id {args.DMS_id} in reference file")
        offset = get_mut_offset_from_reference(args.DMS_id, args.DMS_reference_df, default_offset=1)
        print(f"[debug] WT length={len(wt_seq)}; mutant_col='{args.mutant_col}'; offset={offset}")
        # Compute log-prob matrix for WT using Trainer path to match esm2 infer setup
        pos_to_lp = _score_masked_marginals_with_trainer(wt_seq, args)
        # Build [S, V] tensor; tokenized positions 1..len-2 correspond to aa positions 0..len(wt)-1
        # We'll map by aa index i -> token index i+1
        vocab = next(iter(pos_to_lp.values())).numel() if pos_to_lp else 33
        log_probs_SV = torch.full((len(wt_seq) + 2, vocab), float('-inf'))
        for i, lp_vec in pos_to_lp.items():
            # i corresponds to masked aa index; put at token position i+1
            token_pos = i + 1
            if 0 <= token_pos < log_probs_SV.size(0):
                log_probs_SV[token_pos, : lp_vec.numel()] = lp_vec
        # Map amino acid to token id helper
        def aa_to_id(aa: str) -> int:
            ids = tokenizer.encode(aa, add_special_tokens=False, return_tensors=None)
            return int(ids[0]) if isinstance(ids, list) else int(ids)
        # Positions mapping: tokenized positions 1..len-2 correspond to aa indices 0..len(wt)-1
        def pos_to_token_index(pos_aa: int) -> int:
            return pos_aa + 1
        scores = []
        for mut_spec in DMS_df[args.mutant_col].astype(str).tolist():
            total = 0.0
            for mut in mut_spec.split(":"):
                if not mut:
                    continue
                wt, mt = mut[0], mut[-1]
                idx = int(mut[1:-1]) - offset  # 0-based aa index
                token_pos = pos_to_token_index(idx)
                wt_id = aa_to_id(wt)
                mt_id = aa_to_id(mt)
                lp_vec = log_probs_SV[token_pos]
                total += (lp_vec[mt_id] - lp_vec[wt_id]).item()
            scores.append(total)
        result_df = pd.DataFrame({"log_likelihood": scores}, index=DMS_df.index)
    else:
        raise ValueError(f"Unknown scoring strategy: {args.scoring_strategy}")
    # Save
    output_path = get_likelihood_results_path(args)
    try:
        result_df.to_csv(output_path)
        print(f"Saved likelihoods to: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to save likelihoods to {output_path}: {e}")
    return result_df


def process_single_dataset(DMS_df: pd.DataFrame, args, DMS_reference_df: pd.DataFrame) -> None:
    if hasattr(args, 'down_sample') and args.down_sample is not None:
        DMS_df = stratified_sample_dms(
            DMS_df,
            args.down_sample,
            stratify_column=args.DMS_binary_score_column,
            random_state=42,
        )
    # Attach reference df to args for helpers
    args.DMS_reference_df = DMS_reference_df
    model_scores_df = get_esm2_likelihoods(DMS_df, args, mutation_col=args.sequence_column)
    merged_df = pd.concat([DMS_df, model_scores_df], axis=1)
    performance_results = get_performance_results(
        merged_df,
        args.DMS_score_column,
        args.model_score_column,
        args.DMS_binary_score_column,
    )
    performance_df = pd.DataFrame([performance_results])
    performance_df.index = [args.DMS_id]
    try:
        DMS_reference_entry = DMS_reference_df[DMS_reference_df['DMS_id'] == args.DMS_id]
        if len(DMS_reference_entry) > 0:
            for col in DMS_reference_entry.columns:
                value_series = DMS_reference_entry[col]
                performance_df[col] = value_series.tolist()[0]
    except Exception as e:
        print(f"Warning: Error adding reference data: {e}")
    fitness_output_path = get_fitness_results_path(args)
    try:
        performance_df.to_csv(fitness_output_path, index=False)
        print(f"Saved fitness results to: {fitness_output_path}")
    except Exception as e:
        print(f"Error saving fitness results: {e}")


def process_with_timing(func, *args, **kwargs) -> bool:
    start_time = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        print("------ Full traceback below ------")
        traceback.print_exc()
        # Try to print some quick context if available
        try:
            dms_df = args[0] if len(args) > 0 and isinstance(args[0], pd.DataFrame) else None
            run_args_obj = kwargs.get('args', (args[1] if len(args) > 1 else None))
            ctx_lines = []
            if dms_df is not None:
                ctx_lines.append(f"[context] DMS_df.shape={dms_df.shape}; columns={list(dms_df.columns)}")
            if run_args_obj is not None:
                if hasattr(run_args_obj, 'DMS_id'):
                    ctx_lines.append(f"[context] DMS_id={run_args_obj.DMS_id}")
                if hasattr(run_args_obj, 'DMS_path'):
                    ctx_lines.append(f"[context] DMS_path={run_args_obj.DMS_path}")
                if hasattr(run_args_obj, 'ckpt_dir'):
                    ctx_lines.append(f"[context] ckpt_dir={str(run_args_obj.ckpt_dir)}")
                if hasattr(run_args_obj, 'scoring_strategy'):
                    ctx_lines.append(f"[context] scoring_strategy={run_args_obj.scoring_strategy}")
            if ctx_lines:
                print("\n".join(ctx_lines))
        except Exception:
            pass
        return False


def process_single_dms_file(DMS_filename, args, DMS_reference_df):
    cleanup_memory()
    print(f"\n=== Evaluating {DMS_filename} ===")
    args.DMS_path = os.path.join(args.DMS_scores_folder, DMS_filename)
    args.DMS_id = DMS_filename.replace('.csv', '')
    args.taxon = get_taxon_from_reference(args.DMS_id, DMS_reference_df)
    if os.path.exists(get_fitness_results_path(args)):
        print(f"⏭️  Skipping {DMS_filename} (already exists)")
        return True
    try:
        DMS_df = pd.read_csv(args.DMS_path)
        return process_with_timing(process_single_dataset, DMS_df, args, DMS_reference_df)
    except Exception as e:
        print(f"Error loading file {args.DMS_path}: {e}")
        return False


def process_h5_dms_data(h5_path: str, args, DMS_reference_df):
    print(f"\n=== Reading DMS data from HDF5: {h5_path} ===")
    try:
        args.h5_path = h5_path
        df = read_dms_data_from_h5(h5_path)
        for dms_id in df['DMS_dataset'].unique():
            cleanup_memory()
            DMS_df = df[df['DMS_dataset'] == dms_id].copy()
            args.DMS_id = dms_id.replace('.csv', '')
            args.taxon = get_taxon_from_reference(args.DMS_id, DMS_reference_df)
            if os.path.exists(get_fitness_results_path(args)):
                print(f"⏭️  Skipping {args.DMS_id} (already exists)")
                continue
            print(f"\n=== Processing DMS dataset: {args.DMS_id} ===")
            process_with_timing(process_single_dataset, DMS_df, args, DMS_reference_df)
    except Exception as e:
        print(f"Error processing HDF5 file: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESM2 model fitness evaluation')
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="ESM2 checkpoint directory for inference, or a BioNeMo model tag (e.g., 'esm2/8m:2.0', 'esm2/nv_650m:2.1').",
    )
    # Inference/parallelism args (aligned with ESM2 infer)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor model parallel size.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Pipeline model parallel size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Micro-batch size for prediction.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["fp32", "bf16-mixed"],
        help="Precision for inference.",
    )
    parser.add_argument(
        "--prediction-interval",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="Intervals to write predictions to disk.",
    )
    parser.add_argument(
        "--min-seq-length",
        type=int,
        default=1024,
        help="Minimum sequence length for padding.",
    )
    # DMS data args
    parser.add_argument('--DMS_filenames', type=str, help='Path to text list or CSV of DMS filenames')
    parser.add_argument('--DMS_filenames_column', type=str, default='csv_filename', help='Column name when DMS_filenames is a CSV')
    parser.add_argument('--DMS_path', type=str, help='Path to single DMS file')
    parser.add_argument('--DMS_scores_folder', default='data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides', type=str, help='Path to folder of DMS assays')
    parser.add_argument('--output_performance_file_folder', default='results', type=str, help='Folder to save performance analysis')
    parser.add_argument('--DMS_reference_file_path', default='data/eval_dataset/DMS_ProteinGym_substitutions/DMS_substitutions.csv', type=str, help='Path to DMS reference file')
    parser.add_argument('--DMS_score_column', default='DMS_score', type=str, help='Name of DMS score column')
    parser.add_argument('--DMS_binary_score_column', default='DMS_score_bin', type=str, help='Name of DMS binary score column')
    parser.add_argument('--model_score_column', default='log_likelihood', type=str, help='Name of model score column')
    parser.add_argument('--reduce_method', default='sum', choices=['sum', 'mean'], help='Likelihood reduction method')
    parser.add_argument('--down_sample', type=int, default=None, help='Down-sample size')
    parser.add_argument('--DMS_h5', type=str, help='Path to HDF5 file containing DMS data')
    parser.add_argument('--sequence_column', type=str, default='mutated_sequence', help='Sequence column to score (ESM2 expects protein sequences)')
    parser.add_argument('--scoring_strategy', type=str, default='masked-marginals', choices=['masked-marginals', 'pseudo-ppl'], help='ESM2 scoring strategy')
    parser.add_argument('--mutant_col', type=str, default='mutant', help='Column containing mutation specs like AiB or AiB:CjD')

    args = parser.parse_args()

    # Normalize/resolve checkpoint directory: accept either a real directory path or a BioNeMo model tag
    try:
        if not args.ckpt_dir.is_dir():
            from bionemo.core.data.load import load as bionemo_load  # lazy import
            resolved_ckpt = bionemo_load(str(args.ckpt_dir))
            args.ckpt_dir = Path(resolved_ckpt)
            print(f"Resolved checkpoint tag '{str(args.ckpt_dir)}' to '{resolved_ckpt}'")
    except Exception as e:
        raise ValueError(
            f"--ckpt-dir '{str(args.ckpt_dir)}' is neither an existing directory nor a resolvable BioNeMo tag: {e}"
        )

    # Runtime environment debug
    try:
        print(f"[debug] torch.__version__={torch.__version__}")
        print(f"[debug] torch.cuda.is_available()={torch.cuda.is_available()}; device_count={torch.cuda.device_count()}")
        torch_version_cuda = None
        try:
            import torch.version as torch_version
            torch_version_cuda = getattr(torch_version, 'cuda', None)
        except Exception:
            pass
        print(f"[debug] torch.version.cuda={torch_version_cuda}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        if args.ckpt_dir.is_dir():
            try:
                top_files = sorted(os.listdir(args.ckpt_dir))[:10]
                print(f"[debug] ckpt_dir exists; sample entries: {top_files}")
            except Exception:
                pass
    except Exception:
        pass

    # Seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Load reference
    try:
        DMS_reference_df = pd.read_csv(args.DMS_reference_file_path)
        print(f"Loaded DMS reference file with {len(DMS_reference_df)} entries")
    except Exception as e:
        print(f"Warning: Error loading DMS reference file {args.DMS_reference_file_path}: {e}")
        DMS_reference_df = pd.DataFrame({'DMS_id': [], 'taxon': []})

    # Routes
    if args.DMS_filenames:
        try:
            if args.DMS_filenames_column and args.DMS_filenames.endswith('.csv'):
                DMS_filenames = read_dms_filenames_from_csv(args.DMS_filenames, args.DMS_filenames_column)
            else:
                DMS_filenames = read_dms_filenames_from_text(args.DMS_filenames)
            for DMS_filename in DMS_filenames:
                process_single_dms_file(DMS_filename, args, DMS_reference_df)
        except Exception as e:
            print(f"Error processing DMS files: {e}")
            sys.exit(1)
    elif args.DMS_h5:
        try:
            process_h5_dms_data(args.DMS_h5, args, DMS_reference_df)
        except Exception as e:
            print(f"Error processing HDF5 file: {e}")
            sys.exit(1)
    elif args.DMS_path:
        args.DMS_id = os.path.basename(args.DMS_path).replace('.csv', '')
        args.taxon = get_taxon_from_reference(args.DMS_id, DMS_reference_df)
        if os.path.exists(get_fitness_results_path(args)):
            print(f"Skipping {args.DMS_id} because it already exists")
        else:
            try:
                DMS_df = pd.read_csv(args.DMS_path)
                process_with_timing(process_single_dataset, DMS_df, args, DMS_reference_df)
            except Exception as e:
                print(f"Error processing file {args.DMS_path}: {e}")
                sys.exit(1)
    else:
        raise ValueError("Either --DMS_filenames, --DMS_h5, or --DMS_path must be provided")


