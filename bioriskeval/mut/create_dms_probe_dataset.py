#!/usr/bin/env python3
"""
Create DMS probe dataset with per-dataset fixed-size, stratified splits and BioNeMo representations.

- For each DMS dataset, sample exactly k rows (or all rows if < k)
  - Within those k, split into train (≈80%) and val (≈20%, ceiling), stratified by binned DMS_score
  - Test split is the remainder of the dataset (all rows not selected for train/val)
- Save combined split CSVs (train/val/test) at the base of the output_dir, including stratification labels
- For each model directory, save per-layer H5 files and metadata JSONs for train/val/test


Usage:
    python create_dms_probe_dataset.py --checkpoint_path /path/to/model \
                                            --model_size 7b_arc_longcontext \
                                            --output_dir /path/to/probe_datasets \
                                            --k 40 \
                                            --seed 42 \
                                            [--save_test] \
                                            [--test_mode]
"""

import argparse
import logging
import math
import os
import sys
import h5py
import json
import random
from typing import cast, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# BioNeMo imports
import nemo.lightning as nl
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from megatron.core import parallel_state

# Import from eval_ppl.py like other scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gen'))
from eval_ppl import HyenaPredictor, hyena_predict_forward_step, hyena_predict_data_step  # type: ignore

from utils import (
    extract_model_name,
    check_h5_completion,
    get_existing_sequence_keys,
    build_sequence_key,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_representation_dim(model_size: str) -> int:
    if "40b" in model_size.lower():
        return 8192
    return 4096


def is_rank_zero():
    """Safely determine rank-0.

    Falls back to True when distributed/megatron parallel groups are not initialized
    (e.g., test_mode or CPU-only runs).
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return (
                parallel_state.get_tensor_model_parallel_rank() == 0
                and parallel_state.get_pipeline_model_parallel_rank() == 0
                and parallel_state.get_data_parallel_rank() == 0
            )
        return True
    except Exception:
        return True


class RepresentationExtractor:
    """Extract last-position representations using the proven hook logic."""

    def __init__(self, model, layer_names=["decoder.layers.31"]):
        self.model = model
        self.layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
        self.features: Dict[str, List[torch.Tensor]] = {}
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}

    def _create_hook(self, layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                if len(output.shape) == 3:
                    last_pos_repr = output[-1, :, :].detach().clone()
                else:
                    last_pos_repr = output.detach().clone()
                self.features.setdefault(layer_name, []).append(last_pos_repr)
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                if len(output[0].shape) == 3:
                    last_pos_repr = output[0][-1, :, :].detach().clone()
                else:
                    last_pos_repr = output[0].detach().clone()
                self.features.setdefault(layer_name, []).append(last_pos_repr)
        return hook_fn

    def register_hooks(self) -> None:
        actual_model = self.model
        wrapper_attrs = ['module', 'model', '_model']
        for _ in range(5):
            found = False
            for attr in wrapper_attrs:
                if hasattr(actual_model, attr):
                    nxt = getattr(actual_model, attr)
                    if len(list(nxt.named_children())) > 0:
                        actual_model = nxt
                        found = True
                        break
            if not found:
                break

        if hasattr(actual_model, 'decoder'):
            decoder = actual_model.decoder
        else:
            for attr_name in ['transformer', 'layers', 'blocks']:
                if hasattr(actual_model, attr_name):
                    decoder = getattr(actual_model, attr_name)
                    break
            else:
                raise AttributeError(f"Cannot find decoder/transformer layers in {type(actual_model)}")

        for layer_name in self.layer_names:
            parts = layer_name.split('.')
            current = decoder
            path_parts = parts[1:] if parts[0] == 'decoder' else parts
            for part in path_parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            hook = current.register_forward_hook(self._create_hook(layer_name))
            self.hooks[layer_name] = hook

    def cleanup(self) -> None:
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.features.clear()


class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.text_to_ids(sequence)
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        position_ids = torch.arange(len(tokens_t), dtype=torch.long)
        loss_mask = torch.ones(len(tokens_t), dtype=torch.long)
        return {
            "tokens": tokens_t,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "seq_idx": torch.tensor(idx, dtype=torch.long),
        }


class SequenceDataModule(LightningDataModule):
    def __init__(self, sequences, tokenizer, batch_size=8):
        super().__init__()
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = SequenceDataset(sequences, tokenizer)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        max_len = max(len(item["tokens"]) for item in batch)
        for item in batch:
            seq_len = len(item["tokens"])
            if seq_len < max_len:
                pad_len = max_len - seq_len
                padding = torch.full((pad_len,), self.tokenizer.pad_id, dtype=item["tokens"].dtype)
                item["tokens"] = torch.cat([item["tokens"], padding])
                item["position_ids"] = torch.arange(max_len, dtype=item["position_ids"].dtype)
                mask_padding = torch.zeros(pad_len, dtype=item["loss_mask"].dtype)
                item["loss_mask"] = torch.cat([item["loss_mask"], mask_padding])
        return torch.utils.data.default_collate(batch)


class SimpleInspectionDataset(Dataset):
    """Simple dataset for model warm-up/initialization."""

    def __init__(self, tokenizer, num_samples=2):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"][:num_samples]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.text_to_ids(sequence)
        if len(tokens) < 10:
            tokens = tokens + [self.tokenizer.pad_id] * (10 - len(tokens))
        tokens = torch.tensor(tokens[:50], dtype=torch.long)
        position_ids = torch.arange(len(tokens), dtype=torch.long)
        loss_mask = torch.ones(len(tokens), dtype=torch.long)
        return {
            "tokens": tokens,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "seq_idx": torch.tensor(idx, dtype=torch.long),
        }


class SimpleInspectionDataModule(LightningDataModule):
    """Simple data module for model warm-up/initialization."""

    def __init__(self, tokenizer, batch_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = SimpleInspectionDataset(tokenizer)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        max_len = max(len(item["tokens"]) for item in batch)
        for item in batch:
            seq_len = len(item["tokens"]) 
            if seq_len < max_len:
                pad_len = max_len - seq_len
                padding = torch.full((pad_len,), self.tokenizer.pad_id, dtype=item["tokens"].dtype)
                item["tokens"] = torch.cat([item["tokens"], padding])
                item["position_ids"] = torch.arange(max_len, dtype=item["position_ids"].dtype)
                mask_padding = torch.zeros(pad_len, dtype=item["loss_mask"].dtype)
                item["loss_mask"] = torch.cat([item["loss_mask"], mask_padding])
        return torch.utils.data.default_collate(batch)


def extract_representations(model, tokenizer, trainer, sequences, layer_names, batch_size=8):
    layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
    extractor = RepresentationExtractor(model, layer_names)
    try:
        extractor.register_hooks()
        datamodule = SequenceDataModule(sequences, tokenizer, batch_size=batch_size)
        all_reps = {layer_name: [] for layer_name in layer_names}
        extractor.features.clear()
        _ = trainer.predict(model, datamodule=datamodule)
        captured = extractor.features.copy()
        final: Dict[str, Optional[np.ndarray]] = {}
        for layer_name in layer_names:
            if layer_name in captured and captured[layer_name]:
                tensors = []
                for t in captured[layer_name]:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t.float().cpu().numpy())
                if tensors:
                    arr = np.concatenate(tensors, axis=0)
                    all_reps[layer_name].append(arr)
            if all_reps[layer_name]:
                final[layer_name] = np.concatenate(all_reps[layer_name], axis=0)
            else:
                final[layer_name] = None
        return final
    except Exception as e:
        logger.error(f"Error during representation extraction: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        extractor.cleanup()


def parse_layer_info(layer_name: str) -> str:
    parts = layer_name.split('.')
    layer_num = None
    layer_type = None
    for i, part in enumerate(parts):
        if part.isdigit():
            layer_num = part
            if i + 1 < len(parts):
                layer_type = parts[i + 1]
            break
    if layer_num and layer_type:
        return f"_layer_{layer_num}_{layer_type}"
    elif layer_num:
        return f"_layer_{layer_num}"
    else:
        return f"_{parts[-1]}" if parts else ""


def initialize_hdf5_file(output_path: str, metadata: Dict, representation_dim: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('sequences', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset('representations', (0, representation_dim), maxshape=(None, representation_dim), dtype='float32', compression='gzip')
        f.create_dataset('labels', (0,), maxshape=(None,), dtype='int64')
        f.create_dataset('scores', (0,), maxshape=(None,), dtype='float32')
        f.create_dataset('DMS_dataset', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        for key, value in metadata.items():
            f.attrs[key] = value


def append_to_hdf5(output_path: str, sequences, representations: np.ndarray, labels, scores, dataset_name: str, representation_dim: int) -> None:
    with h5py.File(output_path, 'a') as f:
        seq_ds = cast(h5py.Dataset, f['sequences'])
        rep_ds = cast(h5py.Dataset, f['representations'])
        lab_ds = cast(h5py.Dataset, f['labels'])
        sco_ds = cast(h5py.Dataset, f['scores'])
        dname_ds = cast(h5py.Dataset, f['DMS_dataset'])

        current_size = int(seq_ds.shape[0])
        actual_count = int(representations.shape[0])
        if actual_count == 0:
            return
        new_size = current_size + actual_count

        seq_ds.resize((new_size,))
        rep_ds.resize((new_size, representation_dim))
        lab_ds.resize((new_size,))
        sco_ds.resize((new_size,))
        dname_ds.resize((new_size,))

        seq_ds[current_size:new_size] = sequences[:actual_count]
        rep_ds[current_size:new_size] = representations
        lab_ds[current_size:new_size] = labels[:actual_count]
        sco_ds[current_size:new_size] = np.asarray(scores[:actual_count], dtype=np.float32)
        dname_ds[current_size:new_size] = [dataset_name] * actual_count


def _get_world_size_default_1() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def _pad_to_multiple(sequences, labels, multiple: int, pad_token: str = "A"):
    orig_len = len(sequences)
    if multiple <= 0:
        return sequences, labels, orig_len, 0
    needed = (-orig_len) % multiple
    if needed == 0:
        return sequences, labels, orig_len, 0
    padded_sequences = list(sequences) + [pad_token] * needed
    padded_labels = list(labels) + ([-1] * needed)
    return padded_sequences, padded_labels, orig_len, needed


def make_stratify_labels_from_scores(scores: pd.Series) -> Optional[pd.Series]:
    # Try to bin continuous scores into quantiles for stratified sampling
    q = min(10, max(2, len(scores) // 50))
    while q >= 2:
        try:
            binned = pd.qcut(scores, q=q, duplicates='drop')
        except ValueError:
            q -= 1
            continue
        # Ensure Series for value_counts
        vc = pd.Series(binned).value_counts()
        if vc.shape[0] >= 2 and vc.min() >= 2:
            return pd.Series(binned)
        q -= 1
    return None


def stratified_sample_train_val(df: pd.DataFrame, k: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    # Cap k by dataset size
    k_used = min(k, len(df))
    if k_used <= 0:
        return df.iloc[[]], df.iloc[[]], df.copy(), None

    # Split counts
    val_count = int(math.ceil(0.2 * k_used))
    train_count = int(k_used - val_count)
    if train_count <= 0:
        # if k_used == 1, put into val
        val_count = k_used
        train_count = 0

    # Build stratification labels from DMS_score
    scores: pd.Series = df['DMS_score']  # type: ignore[assignment]
    stratify_labels = make_stratify_labels_from_scores(scores)

    # Choose indices for train/val
    idx = np.arange(len(df))
    try:
        train_idx, val_idx, _, _ = train_test_split(
            idx,
            idx,  # dummy y but length aligned; we use stratify_labels separately
            train_size=train_count,
            test_size=val_count,
            random_state=seed,
            stratify=stratify_labels if isinstance(stratify_labels, pd.Series) else None,
        )
    except Exception:
        # Fallback: random split without stratification
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        selected = idx[:k_used]
        train_idx = selected[:train_count]
        val_idx = selected[train_count:]

    selected_mask = np.zeros(len(df), dtype=bool)
    selected_mask[train_idx] = True
    selected_mask[val_idx] = True

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[~selected_mask].copy()

    # Attach stratification labels for rows to be saved out
    if stratify_labels is not None:
        train_df['stratify_label'] = stratify_labels.iloc[train_idx].astype(str).values
        val_df['stratify_label'] = stratify_labels.iloc[val_idx].astype(str).values
        test_df['stratify_label'] = stratify_labels.iloc[~selected_mask].astype(str).values
    else:
        train_df['stratify_label'] = None
        val_df['stratify_label'] = None
        test_df['stratify_label'] = None

    return train_df, val_df, test_df, stratify_labels


def create_probe_dataset_final(args) -> None:
    seed_everything(args.seed)

    # Paths
    model_name = extract_model_name(args.checkpoint_path)
    # Run-scoped root: include k and seed (and _test if requested)
    run_suffix = f"k={args.k}_seed={args.seed}" 
    run_output_dir = os.path.join(args.output_dir, run_suffix)
    os.makedirs(run_output_dir, exist_ok=True)
    model_output_dir = os.path.join(run_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    ref_csv_path = "/workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv"
    nucleotides_dir = "/workspaces/bionemo-framework/ft-attack/data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides"

    ref_df = pd.read_csv(ref_csv_path)
    logger.info(f"Found {len(ref_df)} datasets in reference file")
    missing_datasets: List[str] = []
    excluded_small_datasets: List[str] = []

    # Determine if we can reuse existing splits (CSV files present for this k/seed run)
    split_paths = {
        'train': os.path.join(run_output_dir, 'dms_split_train.csv'),
        'val': os.path.join(run_output_dir, 'dms_split_val.csv'),
        'test': os.path.join(run_output_dir, 'dms_split_test.csv'),
    }
    reuse_splits = (not args.test_mode) and all(os.path.exists(p) for p in split_paths.values())
    if reuse_splits:
        logger.info("Reusing existing splits from CSV files (k/seed matched by run folder name)")

    representation_dim = get_representation_dim(args.model_size)
    # Determine which splits to save H5s for: default train/val; if --save_test, save test only
    allowed_splits = ['test'] if getattr(args, 'save_test', False) else ['train', 'val']
    # Only initialize tokenizer/model/trainer and H5 structure if not in test mode
    tokenizer = None
    trainer = None
    model = None
    hdf5_files: Dict[Tuple[str, str], str] = {}
    h5_completion: Dict[Tuple[str, str], Dict] = {}
    # Prepare HDF5 info per layer and split (read-only status always; init only when not in test mode)
    for layer_name in args.layer_names:
        for split in allowed_splits:
            layer_info = parse_layer_info(layer_name)
            h5_name = f'dms_probe_dataset{layer_info}_{split}.h5'
            h5_path = os.path.join(model_output_dir, h5_name)
            # Record completion status (safe on all ranks; just reading)
            try:
                status = check_h5_completion(h5_path, run_output_dir)
            except Exception:
                status = {'status': 'mismatch'}
            h5_completion[(layer_name, split)] = status
            # Only initialize file when not in test mode
            if (not args.test_mode) and is_rank_zero():
                if not os.path.exists(h5_path):
                    metadata = {
                        'model_size': args.model_size,
                        'layer_name': layer_name,
                        'layer_names': args.layer_names,
                        'split': split,
                        'k_per_dataset': args.k,
                        'seed': args.seed,
                        'representation_dim': representation_dim,
                    }
                    initialize_hdf5_file(h5_path, metadata, representation_dim)
                hdf5_files[(layer_name, split)] = h5_path

    # In test mode, print a dry-run summary and avoid heavy work
    if args.test_mode and is_rank_zero():
        # Ensure visibility regardless of logger level
        print("[DRY-RUN] test_mode=True | save_test=" + str(getattr(args, 'save_test', False)) +
              " | allowed_splits=" + ",".join(allowed_splits))
        logger.info("Dry-run: H5 completion status summary (no extraction will run)")
        for (layer_name, split), info in sorted(h5_completion.items(), key=lambda x: (x[0][1], x[0][0])):
            layer_info = parse_layer_info(layer_name)
            h5_name = f'dms_probe_dataset{layer_info}_{split}.h5'
            status = str(info.get('status'))
            current_len = int(info.get('current_len', 0))
            expected_len = int(info.get('expected_len', 0))
            action = 'SKIP' if status == 'finished' else ('RESUME' if status == 'partial' else ('CREATE' if status == 'new' else 'CHECK'))
            logger.info(f"{h5_name} | split={split} | status={status} | current={current_len} | expected={expected_len} | action={action}")
            print(f"[DRY-RUN] {h5_name} | split={split} | status={status} | current={current_len} | expected={expected_len} | action={action}")

    # If not test mode and all requested layers/splits are finished, skip the entire model
    if not args.test_mode:
        all_finished = True
        for ln in args.layer_names:
            for sp in allowed_splits:
                if h5_completion.get((ln, sp), {}).get('status') != 'finished':
                    all_finished = False
                    break
            if not all_finished:
                break
        if all_finished:
            if is_rank_zero():
                logger.info("All requested layers/splits already finished; skipping model initialization and extraction.")
            return

    if not args.test_mode:
        tokenizer = get_nmt_tokenizer("byte-level")
        config = HYENA_MODEL_OPTIONS[args.model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            distribute_saved_activations=True,
        )
        model = HyenaPredictor(config, tokenizer=tokenizer)
        seq_len = 8192 if "arc_longcontext" not in args.model_size else 1000000

        trainer = nl.Trainer(
            devices=args.tensor_parallel_size,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy=nl.MegatronStrategy(
                drop_last_batch=False,
                tensor_model_parallel_size=args.tensor_parallel_size,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                ckpt_load_optimizer=False,
                ckpt_save_optimizer=False,
                ckpt_async_save=False,
                sequence_parallel=False,
                ckpt_load_strictness=None,
                data_sampler=nl.MegatronDataSampler(
                    micro_batch_size=args.batch_size,
                    global_batch_size=args.batch_size,
                    seq_len=seq_len,
                    output_log=False,
                ),
            ),
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
            plugins=nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
            ),
        )

        resume = nl.AutoResume(
            resume_if_exists=False,
            resume_ignore_no_checkpoint=False,
            resume_past_end=False,
            restore_config=nl.RestoreConfig(
                path=str(args.checkpoint_path),
                load_model_state=True,
                load_optim_state=False,
                load_artifacts=True,
            ),
        )
        trainer.strategy._setup_optimizers = False  # type: ignore
        resume.setup(trainer, model)

        try:
            dummy_dm = SimpleInspectionDataModule(tokenizer, batch_size=1)
            _ = trainer.predict(model, datamodule=dummy_dm)
        except Exception as e:
            logger.warning(f"Initialization predict failed: {e}")

    # Accumulate combined split CSV rows
    combined_rows: Dict[str, List[Dict]] = {'train': [], 'val': [], 'test': []}
    processed_datasets: List[Dict] = []

    if reuse_splits:
        # Load existing combined CSVs
        df_train = pd.read_csv(split_paths['train'])
        df_val = pd.read_csv(split_paths['val'])
        df_test = pd.read_csv(split_paths['test'])

        # Fill combined_rows
        combined_rows['train'] = df_train.to_dict(orient='records')
        combined_rows['val'] = df_val.to_dict(orient='records')
        combined_rows['test'] = df_test.to_dict(orient='records')

        # Build processed_datasets counts from CSVs
        all_df = pd.concat([df_train.assign(split='train'), df_val.assign(split='val'), df_test.assign(split='test')], ignore_index=True)
        by_ds = all_df.groupby(['dataset_name', 'split']).size().unstack(fill_value=0)
        for dataset_name, counts_row in by_ds.iterrows():
            counts_dict = {k: int(v) for k, v in counts_row.to_dict().items()}
            processed_datasets.append({
                'name': dataset_name,
                'train_count': counts_dict.get('train', 0),
                'val_count': counts_dict.get('val', 0),
                'test_count': counts_dict.get('test', 0),
                'total_rows': int(sum(counts_dict.values())),
            })

        # If not test_mode, generate representations using loaded splits
        if not args.test_mode:
            for dataset_name in by_ds.index.tolist():
                for split_name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
                    if split_name not in allowed_splits:
                        continue
                    # If all layer files for this split are already finished, skip entirely
                    try:
                        if all(h5_completion.get((ln, split_name), {}).get('status') == 'finished' for ln in args.layer_names):
                            continue
                    except Exception:
                        pass
                    ds_split_df = split_df[split_df['dataset_name'] == dataset_name]
                    if ds_split_df.empty:
                        continue
                    seqs = [str(x) for x in ds_split_df['nucleotide_sequence']]
                    labels = [int(x) for x in ds_split_df['DMS_score_bin']]
                    scores = [float(x) for x in ds_split_df['DMS_score']]
                    world_size = _get_world_size_default_1()
                    target_multiple = max(1, args.batch_size) * max(1, world_size)
                    padded_sequences, padded_labels, orig_len, pad_added = _pad_to_multiple(
                        seqs, labels, target_multiple, pad_token="A"
                    )
                    # Extract only for unfinished layers for this split
                    active_layers = [ln for ln in args.layer_names if h5_completion.get((ln, split_name), {}).get('status') != 'finished']
                    if not active_layers:
                        continue
                    reps = extract_representations(model, tokenizer, trainer, padded_sequences, active_layers, args.batch_size)
                    if not reps or all(v is None for v in reps.values()):
                        logger.error(f"Failed to extract representations for {dataset_name} split {split_name}")
                        continue
                    for layer_name, layer_reps in reps.items():
                        if layer_reps is None:
                            continue
                        real_reps = layer_reps[:orig_len]
                        if is_rank_zero():
                            h5_path = hdf5_files.get((layer_name, split_name))
                            if h5_path is None:
                                layer_info = parse_layer_info(layer_name)
                                h5_path = os.path.join(model_output_dir, f'dms_probe_dataset{layer_info}_{split_name}.h5')
                                # Only initialize if not present
                                if not os.path.exists(h5_path):
                                    metadata = {
                                        'model_size': args.model_size,
                                        'layer_name': layer_name,
                                        'layer_names': args.layer_names,
                                        'split': split_name,
                                        'k_per_dataset': args.k,
                                        'seed': args.seed,
                                        'representation_dim': representation_dim,
                                    }
                                    initialize_hdf5_file(h5_path, metadata, representation_dim)
                                hdf5_files[(layer_name, split_name)] = h5_path
                            # Filter already-saved rows to support resume
                            try:
                                existing_keys = get_existing_sequence_keys(h5_path)
                            except Exception:
                                existing_keys = set()
                            miss_idx = [i for i, s in enumerate(seqs) if build_sequence_key(str(dataset_name), s) not in existing_keys]
                            if not miss_idx:
                                continue
                            seqs_to_add = [seqs[i] for i in miss_idx]
                            labels_to_add = [labels[i] for i in miss_idx]
                            scores_to_add = [scores[i] for i in miss_idx]
                            reps_to_add = real_reps[miss_idx]
                            append_to_hdf5(h5_path, seqs_to_add, reps_to_add, labels_to_add, scores_to_add, str(dataset_name), representation_dim)
    else:
        for dataset_idx, (_, row) in enumerate(ref_df.iterrows()):
            csv_filename = str(row['csv_filename'])
            dataset_path = os.path.join(nucleotides_dir, csv_filename)
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset not found: {dataset_path}")
                missing_datasets.append(csv_filename)
                continue

            df = pd.read_csv(dataset_path)
            required_cols = ['nucleotide_sequence', 'DMS_score', 'DMS_score_bin']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.warning(f"Missing required columns in {csv_filename}: {missing}")
                continue

            # Exclude dataset entirely if it has fewer rows than k
            if len(df) < int(args.k):
                logger.warning(f"Excluding dataset for insufficient rows (< k): {csv_filename} | rows={len(df)} < k={args.k}")
                excluded_small_datasets.append(csv_filename)
                continue

            # Add optional protein columns if present
            opt_cols = []
            for c in ['mutant', 'mutated_sequence']:
                if c in df.columns:
                    opt_cols.append(c)

            train_df, val_df, test_df, stratify_labels = stratified_sample_train_val(df, args.k, args.seed + dataset_idx)

            # Record counts
            processed_datasets.append({
                'name': csv_filename,
                'train_count': len(train_df),
                'val_count': len(val_df),
                'test_count': len(test_df),
                'total_rows': len(df),
            })

            # Append to combined CSV rows
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                for _, r in split_df.iterrows():
                    row_dict = {
                        'dataset_name': csv_filename,
                        'nucleotide_sequence': r['nucleotide_sequence'],
                        'DMS_score': r['DMS_score'],
                        'DMS_score_bin': r['DMS_score_bin'],
                        'split': split_name,
                        'stratify_label': r.get('stratify_label', None),
                    }
                    for c in opt_cols:
                        row_dict[c] = r.get(c, None)
                    combined_rows[split_name].append(row_dict)

            # If test_mode, stop after the first dataset that yields any rows
            # if args.test_mode and (len(train_df) + len(val_df) + len(test_df)) > 0:
            #     break

        # Representations and H5 writing only in non-test mode and when not reusing splits
        if (not args.test_mode) and (not reuse_splits):
            split_data = {
                'train': ([str(x) for x in train_df['nucleotide_sequence']], [int(x) for x in train_df['DMS_score_bin']], [float(x) for x in train_df['DMS_score']]),
                'val': ([str(x) for x in val_df['nucleotide_sequence']], [int(x) for x in val_df['DMS_score_bin']], [float(x) for x in val_df['DMS_score']]),
                'test': ([str(x) for x in test_df['nucleotide_sequence']], [int(x) for x in test_df['DMS_score_bin']], [float(x) for x in test_df['DMS_score']]),
            }

            for split, (seqs, labels, scores) in split_data.items():
                if split not in allowed_splits:
                    continue
                if len(seqs) == 0:
                    continue
                world_size = _get_world_size_default_1()
                target_multiple = max(1, args.batch_size) * max(1, world_size)
                padded_sequences, padded_labels, orig_len, pad_added = _pad_to_multiple(
                    seqs, labels, target_multiple, pad_token="A"
                )
                # Extract only for unfinished layers for this split
                active_layers = [ln for ln in args.layer_names if h5_completion.get((ln, split), {}).get('status') != 'finished']
                if not active_layers:
                    continue
                reps = extract_representations(model, tokenizer, trainer, padded_sequences, active_layers, args.batch_size)
                if not reps or all(v is None for v in reps.values()):
                    logger.error(f"Failed to extract representations for {csv_filename} split {split}")
                    continue
                for layer_name, layer_reps in reps.items():
                    if layer_reps is None:
                        continue
                    real_reps = layer_reps[:orig_len]
                    if is_rank_zero():
                        h5_path = hdf5_files.get((layer_name, split))
                        if h5_path is None:
                            layer_info = parse_layer_info(layer_name)
                            h5_path = os.path.join(model_output_dir, f'dms_probe_dataset{layer_info}_{split}.h5')
                            if not os.path.exists(h5_path):
                                metadata = {
                                    'model_size': args.model_size,
                                    'layer_name': layer_name,
                                    'layer_names': args.layer_names,
                                    'split': split,
                                    'k_per_dataset': args.k,
                                    'seed': args.seed,
                                    'representation_dim': representation_dim,
                                }
                                initialize_hdf5_file(h5_path, metadata, representation_dim)
                            hdf5_files[(layer_name, split)] = h5_path
                        # Resume support: filter already saved rows
                        try:
                            existing_keys = get_existing_sequence_keys(h5_path)
                        except Exception:
                            existing_keys = set()
                        miss_idx = [i for i, s in enumerate(seqs) if build_sequence_key(csv_filename, s) not in existing_keys]
                        if not miss_idx:
                            continue
                        seqs_to_add = [seqs[i] for i in miss_idx]
                        labels_to_add = [labels[i] for i in miss_idx]
                        scores_to_add = [scores[i] for i in miss_idx]
                        reps_to_add = real_reps[miss_idx]
                        append_to_hdf5(h5_path, seqs_to_add, reps_to_add, labels_to_add, scores_to_add, csv_filename, representation_dim)

            # test_mode handled via dataset_idx guard above

    # Write combined split CSVs at output_dir root
    if not reuse_splits:
        if is_rank_zero():
            for split in ['train', 'val', 'test']:
                df_out = pd.DataFrame(combined_rows[split])
                out_path = os.path.join(run_output_dir, f'dms_split_{split}.csv')
                df_out.to_csv(out_path, index=False)

    # Write run-level metadata at run_output_dir root (available in both test and non-test runs)
    if is_rank_zero():
        unique_datasets = sorted(list({row['dataset_name'] for s in ['train','val','test'] for row in combined_rows[s]}))
        run_meta = {
            'k_per_dataset': args.k,
            'seed': args.seed,
            'reuse_splits': bool(reuse_splits),
            'reference_total_datasets': int(len(ref_df)),
            'nucleotides_dir': nucleotides_dir,
            'split_counts': {
                'train': int(len(combined_rows['train'])),
                'val': int(len(combined_rows['val'])),
                'test': int(len(combined_rows['test'])),
            },
            'per_dataset_counts': processed_datasets,
            'datasets_in_splits': unique_datasets,
            'missing_datasets': missing_datasets,
            'excluded_datasets_insufficient_rows': excluded_small_datasets,
            'csv_paths': split_paths,
        }
        with open(os.path.join(run_output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(run_meta, f, indent=2)

    # Save metadata per layer and split
    if not args.test_mode:
        saved_files = []
        for layer_name in args.layer_names:
            layer_info = parse_layer_info(layer_name)
            for split in allowed_splits:
                h5_name = f'dms_probe_dataset{layer_info}_{split}.h5'
                saved_files.append(h5_name)
                total_sequences = 0
                key = 'train_count' if split == 'train' else ('val_count' if split == 'val' else 'test_count')
                for d in processed_datasets:
                    total_sequences += int(d[key])
                metadata = {
                    'model_size': args.model_size,
                    'layer_name': layer_name,
                    'layer_names': args.layer_names,
                    'split': split,
                    'k_per_dataset': args.k,
                    'seed': args.seed,
                    'total_sequences': total_sequences,
                    'representation_dim': representation_dim,
                    'datasets': [d['name'] for d in processed_datasets],
                }
                meta_filename = f'dataset_metadata{layer_info}_{split}.json'
                meta_path = os.path.join(model_output_dir, meta_filename)
                if is_rank_zero():
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

        # Overall metadata
        overall = {
            'model_size': args.model_size,
            'layer_names': args.layer_names,
            'k_per_dataset': args.k,
            'seed': args.seed,
            'total_datasets': len(processed_datasets),
            'total_train_sequences': sum(d['train_count'] for d in processed_datasets),
            'total_val_sequences': sum(d['val_count'] for d in processed_datasets),
            'total_test_sequences': sum(d['test_count'] for d in processed_datasets),
            'datasets': [d['name'] for d in processed_datasets],
            'saved_files': saved_files,
            'combined_split_csvs': [f'dms_split_{s}.csv' for s in ['train', 'val', 'test']],
        }
        overall_path = os.path.join(model_output_dir, 'overall_metadata.json')
        if is_rank_zero():
            with open(overall_path, 'w') as f:
                json.dump(overall, f, indent=2)

    logger.info("Dataset creation completed!")
    logger.info(f"Files saved to: {model_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create final DMS probe dataset with fixed per-dataset sampling and BioNeMo representations")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to BioNeMo model checkpoint")
    parser.add_argument("--model_size", type=str, default="7b_arc_longcontext", choices=sorted(HYENA_MODEL_OPTIONS.keys()), help="Model size configuration")
    parser.add_argument("--layer_names", type=str, nargs='+', default=["decoder.layers.26", "decoder.layers.30"], help="Layer names to extract representations from")
    parser.add_argument("--output_dir", type=str, default="/workspaces/bionemo-framework/ft-attack/data/eval_dataset/fitness/probe_datasets_stratified", help="Output directory for datasets")
    parser.add_argument("--k", type=int, default=40, help="Number of samples per dataset for train+val combined")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for representation extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_mode", action="store_true", help="Print representations need to collect, do not run extraction")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--save_test", action="store_true", help="If set, save only test split H5 files instead of train/val")

    args = parser.parse_args()
    logger.info("Starting final DMS probe dataset creation...")
    logger.info(f"Arguments: {vars(args)}")
    try:
        create_probe_dataset_final(args)
        logger.info("Dataset creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


