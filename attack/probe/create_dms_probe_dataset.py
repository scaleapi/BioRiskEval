#!/usr/bin/env python3
"""
Create linear probe dataset from DMS data with Evo2 representations.
 
This script:
1. Reads the reference CSV to get list of DMS datasets
2. For each dataset, samples 500 sequences with balanced sampling
3. Extracts representations using BioNeMo model at the last position
4. Saves dataset efficiently to disk
 
Usage:
    python create_dms_probe_dataset.py --checkpoint_path /path/to/model \
                                      --model_size 7b_arc_longcontext \
                                      --output_dir ./probe_datasets \
                                      --seed 42
"""
 
import argparse
import logging
import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.utils import resample
import json
import warnings
from typing import cast
 
 
# Import BioNeMo components
import nemo.lightning as nl
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from megatron.core import parallel_state
 
# Import from eval_ppl.py like other scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
from eval_ppl import HyenaPredictor, hyena_predict_forward_step, hyena_predict_data_step
 
# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import extract_model_name
 
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore", category=UserWarning)



 
def is_rank_zero():
    return (
        parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_pipeline_model_parallel_rank() == 0
        and parallel_state.get_data_parallel_rank()    == 0
    )
    
 
class RepresentationExtractor:
    """Extract representations from the last sequence position using proven test_hooks.py logic."""
    
    def __init__(self, model, layer_names=["decoder.layers.31"]):
        self.model = model
        self.layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
        self.features = {}
        self.hooks = {}
        # Index tracking not needed; Lightning preserves order with shuffle=False
        
    def _create_hook(self, layer_name):
        """Create hook to capture last position representations."""
        def hook_fn(module, input, output):
            # The hook gets called for every token position
            # We want to keep overwriting until we get the final position
            if isinstance(output, torch.Tensor):
                # Shape is [seq_length, batch, hidden] = [seq_len, 1, 4096]
                # Extract the last sequence position: [seq_len, 1, 4096] -> [1, 4096]
                if len(output.shape) == 3:
                    last_pos_repr = output[-1, :, :].detach().clone()  # Take the last sequence position
                else:
                    # Fallback for unexpected shapes
                    last_pos_repr = output.detach().clone()
                
                # Accumulate representations across all batches
                if layer_name not in self.features:
                    self.features[layer_name] = []
                self.features[layer_name].append(last_pos_repr)
                
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    # Handle tuple outputs the same way - [seq_length, batch, hidden]
                    if len(output[0].shape) == 3:
                        last_pos_repr = output[0][-1, :, :].detach().clone()  # Take the last sequence position
                    else:
                        last_pos_repr = output[0].detach().clone()
                    # Accumulate representations across all batches
                    if layer_name not in self.features:
                        self.features[layer_name] = []
                    self.features[layer_name].append(last_pos_repr)
        return hook_fn
    
    def register_hooks(self):
        """Register hooks using exact test_hooks.py logic."""
        
        # Copy the exact model navigation logic
        actual_model = self.model
        unwrapping_path = []
        
        # Common wrapper patterns in distributed training (from test_hooks.py)
        wrapper_attrs = ['module', 'model', '_model']
        max_unwrap_depth = 5
        
        for depth in range(max_unwrap_depth):
            found_wrapper = False
            for attr in wrapper_attrs:
                if hasattr(actual_model, attr):
                    next_model = getattr(actual_model, attr)
                    # Check if this is actually a wrapper (has fewer direct children than parameters suggest)
                    children = list(next_model.named_children())
                    if len(children) > 0:  # This looks like the actual model
                        unwrapping_path.append(f"{attr}")
                        actual_model = next_model
                        found_wrapper = True
                        print(f"Unwrapped through '{attr}': {type(actual_model).__name__}")
                        break
            
            if not found_wrapper:
                break
        
        if unwrapping_path:
            print(f"Model unwrapping path: {' -> '.join(unwrapping_path)}")
            print(f"Final unwrapped model: {type(actual_model).__name__}")
        
        # Navigate to the decoder layers (exact logic from test_hooks.py)
        if hasattr(actual_model, 'decoder'):
            decoder = actual_model.decoder
            print(f"Found decoder: {type(decoder).__name__}")
        else:
            # Try alternative paths
            for attr_name in ['transformer', 'layers', 'blocks']:
                if hasattr(actual_model, attr_name):
                    decoder = getattr(actual_model, attr_name)
                    print(f"Found decoder via {attr_name}: {type(decoder).__name__}")
                    break
            else:
                # List available attributes for debugging
                attrs = [attr for attr in dir(actual_model) if not attr.startswith('_')]
                logger.error(f"Cannot find decoder/transformer layers in {type(actual_model)}")
                logger.error(f"Available attributes: {attrs[:10]}...")  # Show first 10
                raise AttributeError(f"Cannot find decoder/transformer layers in {type(actual_model)}")
        
        # Register hooks on target layers (exact logic from test_hooks.py)
        for layer_name in self.layer_names:
            print(f"Attempting to register hook on: {layer_name}")
            
            # Parse layer path (e.g., "decoder.layers.26.mlp")
            parts = layer_name.split('.')
            
            # Start from decoder
            current = decoder
            
            # Navigate to the target layer (skip 'decoder' since we already have it)
            path_parts = parts[1:] if parts[0] == 'decoder' else parts
            
            for i, part in enumerate(path_parts):
                try:
                    if part.isdigit():
                        current = current[int(part)]
                        print(f"  Navigated to index {part}: {type(current).__name__}")
                    else:
                        current = getattr(current, part)
                        print(f"  Navigated to attribute {part}: {type(current).__name__}")
                except (AttributeError, IndexError, TypeError) as e:
                    logger.error(f"  Failed to navigate to {part} at step {i}: {e}")
                    logger.error(f"  Current object: {type(current).__name__}")
                    if hasattr(current, '__dict__'):
                        attrs = [attr for attr in dir(current) if not attr.startswith('_')]
                        logger.error(f"  Available attributes: {attrs[:10]}...")
                    raise
            
            # Register hook (using our last-position hook, not the full output hook)
            hook = current.register_forward_hook(self._create_hook(layer_name))
            self.hooks[layer_name] = hook
            print(f"✅ Registered hook on {layer_name}")
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.features.clear()
 
 
class SequenceDataset(Dataset):
    """Dataset for representation extraction."""
    
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.text_to_ids(sequence)
        
        # Convert to tensor and create other required inputs
        tokens = torch.tensor(tokens, dtype=torch.long)
        position_ids = torch.arange(len(tokens), dtype=torch.long)
        loss_mask = torch.ones(len(tokens), dtype=torch.long)
        
        return {
            "tokens": tokens,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "seq_idx": torch.tensor(idx, dtype=torch.long),  # Add sequence index
        }
 
 
class SequenceDataModule(LightningDataModule):
    """Data module for representation extraction."""
    
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
        """Collate function with padding."""
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
 


def load_and_sample_dataset(csv_path, n_samples=624, seed=42):
    """Improved version with better balance guarantees"""
    print(f"Loading dataset: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['nucleotide_sequence', 'DMS_score_bin','DMS_score']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns in {csv_path}. Required: {required_cols}")
        return None, None
    
    # Check class distribution
    class_counts = df['DMS_score_bin'].value_counts()
    print(f"Original class distribution: {dict(class_counts)}")
    
    # Determine the largest balanced subset available
    min_class_size = class_counts.min()
    max_balanced_total = min_class_size * 2
    
    # Cap by availability, then split 80/20 per class
    per_class_total = min(min_class_size, n_samples // 2)
    if per_class_total == 0:
        print(f"Insufficient data to create balanced splits (min_class_size={min_class_size})")
        return None, None
    # Use floor for 80% to match prior 249/63 behavior at n_samples=624
    train_per_class = int(per_class_total * 0.8)
    test_per_class = per_class_total - train_per_class
    print(f"Balanced target totals -> Train: {train_per_class * 2}, Test: {test_per_class * 2}")
    
    if train_per_class == 0 or test_per_class == 0:
        print(f"Not enough samples for both train and test in {csv_path}")
        return None, None
    
    # Stratified sampling with guaranteed balance
    np.random.seed(seed)
    train_dfs = []
    test_dfs = []
    
    for class_label in [0, 1]:
        class_df = df[df['DMS_score_bin'] == class_label]
        
        # Ensure we have enough samples for both train and test
        total_needed = train_per_class + test_per_class
        
        if len(class_df) >= total_needed:
            # Sample without replacement, then split
            sampled_df = class_df.sample(n=total_needed, random_state=seed + class_label)
            train_class_df = sampled_df.iloc[:train_per_class]
            test_class_df = sampled_df.iloc[train_per_class:]
        else:
            # Need to oversample
            sampled_df = pd.DataFrame(resample(class_df, n_samples=total_needed, 
                                              random_state=seed + class_label, replace=True))
            train_class_df = sampled_df.iloc[:train_per_class]
            test_class_df = sampled_df.iloc[train_per_class:]
        
        train_dfs.append(train_class_df)
        test_dfs.append(test_class_df)
    
    # Combine and shuffle
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    
    print(f"SAMPLING RESULT - Train dataset: {len(train_df)} samples, {dict(train_df['DMS_score_bin'].value_counts())}")
    print(f"SAMPLING RESULT - Test dataset: {len(test_df)} samples, {dict(test_df['DMS_score_bin'].value_counts())}")
    
    return train_df, test_df

def extract_representations(model, tokenizer, trainer, sequences, layer_names, batch_size=8):
    """Extract representations for a list of sequences from multiple layers."""
    layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
    print(f"Extracting representations for {len(sequences)} sequences from {len(layer_names)} layers")
    
    # Create representation extractor
    extractor = RepresentationExtractor(model, layer_names)
    
    try:
        extractor.register_hooks()
        
        # Create data module with index tracking
        datamodule = SequenceDataModule(sequences, tokenizer, batch_size=batch_size)
        
        # Extract representations in batches - one dict per layer
        all_representations = {layer_name: [] for layer_name in layer_names}
        
        # Extract features using Lightning's predict method (like test_hooks.py)
        print("Running representation extraction via Lightning predict...")
        
        # Clear any previous features
        extractor.features.clear()
        
        # Run prediction - this will trigger our hooks with index tracking
        results = trainer.predict(model, datamodule=datamodule)
        
        # Get the features that were captured by our hooks
        captured_features = extractor.features.copy()
        
        # Convert captured features to the expected format with proper ordering
        if captured_features:
            for layer_name in layer_names:
                if layer_name in captured_features and captured_features[layer_name]:
                    # Features are a list of tensors, one per processed batch
                    batch_list = captured_features[layer_name]
                    if isinstance(batch_list, list) and len(batch_list) > 0:
                        rep_batches = []
                        for item in batch_list:
                            if isinstance(item, torch.Tensor):
                                rep_batches.append(item.float().cpu().numpy())
                        if rep_batches:
                            batch_repr = np.concatenate(rep_batches, axis=0)
                            all_representations[layer_name].append(batch_repr)
                            print(f"Captured {batch_repr.shape[0]} representations for layer {layer_name}")
                        else:
                            logger.warning(f"No tensor features found for layer {layer_name}")
                    else:
                        logger.warning(f"Unexpected format for captured features in layer {layer_name}")
                else:
                    logger.warning(f"No representations captured for layer {layer_name}")
        else:
            logger.error("No features were captured by hooks")
        
        # Concatenate results for each layer
        final_representations = {}
        for layer_name in layer_names:
            if all_representations[layer_name]:
                final_representations[layer_name] = np.concatenate(all_representations[layer_name], axis=0)
                print(f"Extracted representations for {layer_name}: {final_representations[layer_name].shape}")
            else:
                logger.error(f"No representations were extracted for layer {layer_name}")
                final_representations[layer_name] = None
        
        return final_representations
    
    except Exception as e:
        logger.error(f"Error during representation extraction: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        extractor.cleanup()
 
def parse_layer_info(layer_name):
    """Parse layer name to extract layer number and type."""
    parts = layer_name.split('.')
    layer_num = None
    layer_type = None
    
    for i, part in enumerate(parts):
        if part.isdigit():
            layer_num = part
            # Get the next part as the layer type if it exists
            if i + 1 < len(parts):
                layer_type = parts[i + 1]
            break
    
    if layer_num and layer_type:
        return f"_layer_{layer_num}_{layer_type}"
    elif layer_num:
        return f"_layer_{layer_num}"
    else:
        # Fallback: use the last part of the layer name
        return f"_{parts[-1]}" if parts else ""
    
        
def initialize_hdf5_file(output_path, metadata, total_sequences):
    """Initialize HDF5 file with resizable datasets."""
    print(f"Initializing HDF5 file: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create resizable datasets
        f.create_dataset('sequences', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset('representations', (0, 4096), maxshape=(None, 4096), dtype='float32', compression='gzip')
        f.create_dataset('labels', (0,), maxshape=(None,), dtype='int64')
        f.create_dataset('scores', (0,), maxshape=(None,), dtype='float32')
        f.create_dataset('DMS_dataset', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        
        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
    
    print(f"HDF5 file initialized")
 
 
# Helper functions for padding to avoid DDP partial-batch drops

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
 
 
def append_to_hdf5(output_path, sequences, representations, labels, scores, dataset_name):
    """Append data to existing HDF5 file, sized by actually captured representations."""
    import h5py
    with h5py.File(output_path, 'a') as f:
        # Access datasets explicitly with typing casts (satisfy static checker)
        seq_ds = cast(h5py.Dataset, f['sequences'])
        rep_ds = cast(h5py.Dataset, f['representations'])
        lab_ds = cast(h5py.Dataset, f['labels'])
        sco_ds = cast(h5py.Dataset, f['scores'])
        dname_ds = cast(h5py.Dataset, f['DMS_dataset'])

        # Current size from dataset shape
        current_size = int(seq_ds.shape[0])

        # Use actual representation count, not input sequence count
        actual_count = int(representations.shape[0])
        if actual_count == 0:
            print(f"No representations captured; skipping append for {output_path}")
            return

        new_size = current_size + actual_count

        # Resize datasets exactly to the new size
        seq_ds.resize((new_size,))  # type: ignore
        rep_ds.resize((new_size, 4096))  # type: ignore
        lab_ds.resize((new_size,))  # type: ignore
        sco_ds.resize((new_size,))  # type: ignore
        dname_ds.resize((new_size,))  # type: ignore

        # Append only the sequences/labels/scores that actually have representations
        seq_ds[current_size:new_size] = sequences[:actual_count]
        rep_ds[current_size:new_size] = representations
        lab_ds[current_size:new_size] = labels[:actual_count]
        sco_ds[current_size:new_size] = np.asarray(scores[:actual_count], dtype=np.float32)
        dname_ds[current_size:new_size] = [dataset_name] * actual_count
    
    print(f"Appended {actual_count} sequences to {output_path}")
 
 
def create_probe_dataset(args):
    """Main function to create the probe dataset."""
    
    # Set up paths
    model_name = extract_model_name(args.checkpoint_path)
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Using model-specific output directory: {model_output_dir}")
    print(f"Extracted model name: {model_name}")
 
    ref_csv_path = "/workspaces/BioRiskEval/attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv"
    nucleotides_dir = "/workspaces/BioRiskEval/attack/data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides"
 
    ref_df = pd.read_csv(ref_csv_path)
    print(f"Found {len(ref_df)} datasets in reference file")
 
 
    print("Loading tokenizer...")
    tokenizer = get_nmt_tokenizer("byte-level")
 
    # Create model config
    print(f"Creating model config: {args.model_size}")
    config = HYENA_MODEL_OPTIONS[args.model_size](
        forward_step_fn=hyena_predict_forward_step,
        data_step_fn=hyena_predict_data_step,
        distribute_saved_activations=True,
    )
 
    # Create model
    print("Creating model...")
    model = HyenaPredictor(config, tokenizer=tokenizer)

    seq_len = 8192 if "arc_longcontext" not in args.model_size else 1000000

 
    # Create trainer
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
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
    print("Model loaded successfully!")
    
    # Initialize model properly by running a test prediction (like test_hooks.py)
    print("Initializing model with test prediction...")
    try:
        sys.path.append(os.path.dirname(__file__))
        from test_hooks import SimpleInspectionDataModule
        dummy_datamodule = SimpleInspectionDataModule(tokenizer, batch_size=1)
        results = trainer.predict(model, datamodule=dummy_datamodule)
        print(f"✅ Model initialization completed, got {len(results) if results else 0} batches")
 
    except Exception as e:
        logger.warning(f"Test prediction failed: {e}, continuing anyway...")
    
    # Initialize HDF5 files for each layer and split (train/test)
    hdf5_files = {}
    
    model_name = args.checkpoint_path.split("/")[-1]
    
    for layer_name in args.layer_names:
        for split in ['train', 'test']:
            # Create filename with layer and split information
            layer_info = parse_layer_info(layer_name)
            hdf5_filename = f'dms_probe_dataset{layer_info}_{split}.h5'
            hdf5_path = os.path.join(model_output_dir, hdf5_filename)
            
            # Create initial metadata for this layer and split
            metadata = {
                'model_size': args.model_size,
                'layer_name': layer_name,
                'layer_names': args.layer_names,
                'split': split,
                'n_samples_per_dataset': args.n_samples,
                'seed': args.seed,
                'representation_dim': 4096,
            }
            if is_rank_zero():
                initialize_hdf5_file(hdf5_path, metadata, 0)  # We don't know total size yet
                hdf5_files[(layer_name, split)] = hdf5_path
 
    # ------------------------------------------------------------------
    # Process each dataset incrementally  
    # ------------------------------------------------------------------
    
    # Track processed datasets for metadata
    processed_datasets = []
    
    for dataset_idx, (idx, row) in enumerate(ref_df.iterrows()):
        csv_filename = str(row['csv_filename'])
        dataset_path = os.path.join(nucleotides_dir, csv_filename)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        
        print(f"Processing dataset {dataset_idx+1}/{len(ref_df)}: {csv_filename}")
        
        # Load and sample dataset using the fixed function from sampling_test
        train_df, test_df = load_and_sample_dataset(dataset_path, n_samples=args.n_samples, seed=args.seed)
        if train_df is None or test_df is None:
            continue
        
        # Prepare sequences and labels for both splits
        train_sequences = train_df['nucleotide_sequence'].tolist()
        train_labels = train_df['DMS_score_bin'].tolist()
        train_scores = train_df['DMS_score'].tolist()
        test_sequences = test_df['nucleotide_sequence'].tolist()
        test_labels = test_df['DMS_score_bin'].tolist()
        test_scores = test_df['DMS_score'].tolist()
        
        # Track this dataset for metadata
        processed_datasets.append({
            'name': csv_filename,
            'train_count': len(train_sequences),
            'test_count': len(test_sequences)
        })
 
        split_data = {
            'train': (train_sequences, train_labels, train_scores),
            'test': (test_sequences, test_labels, test_scores),
        }
 
        # Extract and save representations per split
        for split, (split_sequences, split_labels, split_scores) in split_data.items():
            print(f"Extracting representations for {split} split: {len(split_sequences)} sequences")

            # Pad to avoid DDP dropping last batches: make divisible by batch_size * world_size
            world_size = _get_world_size_default_1()
            target_multiple = max(1, args.batch_size) * max(1, world_size)
            padded_sequences, padded_labels, orig_len, pad_added = _pad_to_multiple(
                split_sequences, split_labels, target_multiple, pad_token="A"
            )
            if pad_added:
                print(f"Padded {split} split with {pad_added} samples to {len(padded_sequences)} (multiple of {target_multiple})")

            representations_dict = extract_representations(
                model, tokenizer, trainer, padded_sequences, args.layer_names, args.batch_size
            )

            if not representations_dict or all(v is None for v in representations_dict.values()):
                logger.error(f"Failed to extract representations for {split} split")
                continue

            for layer_name, layer_representations in representations_dict.items():
                if is_rank_zero() and layer_representations is not None:
                    # Remove any padded tail before writing
                    real_representations = layer_representations[:orig_len]

                    hdf5_path = hdf5_files[(layer_name, split)]
                    append_to_hdf5(hdf5_path, split_sequences, real_representations, split_labels, split_scores, csv_filename)
 
        if args.test_mode:
            break
 
 
 
    # After processing all datasets, save metadata files
    saved_files = []
    for layer_name in args.layer_names:
        layer_info = parse_layer_info(layer_name)
        
        for split in ['train', 'test']:
            hdf5_filename = f'dms_probe_dataset{layer_info}_{split}.h5'
            saved_files.append(hdf5_filename)
 
            # Save layer-specific metadata as JSON
            total_sequences = sum(dataset[f'{split}_count'] for dataset in processed_datasets)
            metadata = {
                'model_size': args.model_size,
                'layer_name': layer_name,
                'layer_names': args.layer_names,
                'split': split,
                'n_samples_per_dataset': args.n_samples,
                'seed': args.seed,
                'total_sequences': total_sequences,
                'representation_dim': 4096,
                'datasets': [dataset['name'] for dataset in processed_datasets]
            }
            
            metadata_filename = f'dataset_metadata{layer_info}_{split}.json'
            metadata_path = os.path.join(model_output_dir, metadata_filename)
            if is_rank_zero():
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
 
    # Save overall metadata with info about all layers
    total_train_sequences = sum(dataset['train_count'] for dataset in processed_datasets)
    total_test_sequences = sum(dataset['test_count'] for dataset in processed_datasets)
    overall_metadata = {
        'model_size': args.model_size,
        'layer_names': args.layer_names,
        'n_samples_per_dataset': args.n_samples,
        'seed': args.seed,
        'total_train_sequences': total_train_sequences,
        'total_test_sequences': total_test_sequences,
        'total_datasets': len(processed_datasets),
        'datasets': [dataset['name'] for dataset in processed_datasets],
        'saved_files': saved_files
    }
    overall_metadata_path = os.path.join(model_output_dir, 'overall_metadata.json')
    if is_rank_zero():
        with open(overall_metadata_path, 'w') as f:
            json.dump(overall_metadata, f, indent=2)
 
    print(f"Dataset creation completed!")
    print(f"Files saved to: {model_output_dir}")
    for filename in saved_files:
        print(f"  - {filename}: Layer-specific dataset")
    print(f"  - overall_metadata.json: Overall metadata")
    print(f"Processed {len(processed_datasets)} datasets with {total_train_sequences} train and {total_test_sequences} test sequences")
 
 
def main():
    parser = argparse.ArgumentParser(description="Create DMS probe dataset with BioNeMo representations")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to BioNeMo model checkpoint")
    parser.add_argument("--model_size", type=str, default="7b_arc_longcontext",
                       choices=sorted(HYENA_MODEL_OPTIONS.keys()),
                       help="Model size configuration")
    parser.add_argument("--layer_names", type=str, nargs='+', default=["decoder.layers.26", "decoder.layers.30"],
                       help="Layer names to extract representations from (can specify multiple)")
    parser.add_argument("--output_dir", type=str, default="/workspaces/BioRiskEval/attack/data/eval_dataset/fitness/probe_datasets",
                       help="Output directory for datasets")
    parser.add_argument("--n_samples", type=int, default=624,
                       help="Total samples per dataset (will be split into train/test)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for representation extraction")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode: process only the first dataset for quick testing")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    
    args = parser.parse_args()
    
    print("Starting DMS probe dataset creation...")
    print(f"Arguments: {vars(args)}")
    
    try:
        create_probe_dataset(args)
        print("Dataset creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1
 
 
if __name__ == "__main__":
    exit(main()) 