#!/usr/bin/env python3
"""

Create a simple train/test/rest split of DMS datasets for linear probe training, without extracting model representations.

This script:
1. Reads a reference CSV to get the list of DMS datasets.
2. For each dataset, samples a balanced set of train/test examples, or the rest of the data if the rest split is requested.
3. Saves the train and test splits to HDF5 files with the required columns.

Usage:
    python create_dms_dataset_split.py --output_dir /path/to/output \
                                   --n_samples 624 \
                                   --seed 42
"""



import argparse
import logging
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import resample
import json
import warnings
from typing import cast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_sample_dataset(csv_path, n_samples=624, seed=42):
    """Improved version with better balance guarantees (same as original script)"""
    print(f"Loading dataset: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['nucleotide_sequence', 'DMS_score_bin', 'DMS_score', 'mutant']
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

def compute_rest_df(full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Compute REST split = full_df minus the specific train/test selections.

    We remove the exact rows selected for train and test using a multiset keyed by
    stable identifying columns, preserving multiplicities so duplicates are handled.
    """
    required_cols = ['nucleotide_sequence', 'DMS_score_bin', 'DMS_score', 'mutant']
    if not all(col in full_df.columns for col in required_cols):
        return pd.DataFrame(columns=required_cols)

    from collections import Counter

    def row_key(row):
        return (
            row['nucleotide_sequence'],
            int(row['DMS_score_bin']),
            float(row['DMS_score']),
            str(row['mutant']),
        )

    exclude_counter: Counter = Counter()
    for _, r in pd.concat([train_df[required_cols], test_df[required_cols]], ignore_index=True).iterrows():
        exclude_counter[row_key(r)] += 1

    kept_rows = []
    for _, r in full_df.iterrows():
        k = row_key(r)
        if exclude_counter.get(k, 0) > 0:
            exclude_counter[k] -= 1
            continue
        kept_rows.append(r)

    return pd.DataFrame(kept_rows).reset_index(drop=True)

def initialize_hdf5_file(output_path, metadata):
    """Initialize HDF5 file with resizable datasets for the required columns."""
    print(f"Initializing HDF5 file: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create resizable datasets for the required columns
        f.create_dataset('sequences', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset('scores', (0,), maxshape=(None,), dtype='float32')
        f.create_dataset('DMS_dataset', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset('labels', (0,), maxshape=(None,), dtype='int64')
        f.create_dataset('mutant', (0,), maxshape=(None,), dtype=h5py.string_dtype())
        
        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
    
    print(f"HDF5 file initialized")

def append_to_hdf5(output_path, sequences, scores, dms_datasets, labels, mutants):
    """Append data to existing HDF5 file."""
    with h5py.File(output_path, 'a') as f:
        # Access datasets explicitly with typing casts
        seq_ds = cast(h5py.Dataset, f['sequences'])
        sco_ds = cast(h5py.Dataset, f['scores'])
        dms_ds = cast(h5py.Dataset, f['DMS_dataset'])
        lab_ds = cast(h5py.Dataset, f['labels'])
        mut_ds = cast(h5py.Dataset, f['mutant'])

        # Current size from dataset shape
        current_size = int(seq_ds.shape[0])
        new_count = len(sequences)
        new_size = current_size + new_count

        if new_count == 0:
            print(f"No data to append to {output_path}")
            return

        # Resize datasets exactly to the new size
        seq_ds.resize((new_size,))
        sco_ds.resize((new_size,))
        dms_ds.resize((new_size,))
        lab_ds.resize((new_size,))
        mut_ds.resize((new_size,))

        # Append the data
        seq_ds[current_size:new_size] = sequences
        sco_ds[current_size:new_size] = np.asarray(scores, dtype=np.float32)
        dms_ds[current_size:new_size] = dms_datasets
        lab_ds[current_size:new_size] = labels
        mut_ds[current_size:new_size] = mutants
    
    print(f"Appended {new_count} sequences to {output_path}")

def create_probe_datasets(args):
    """Main function to create the probe datasets."""
    
    # Set up paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")

    ref_csv_path = "/workspaces/src/bionemo-framework/attack/data/eval_dataset/DMS_ProteinGym_substitutions/virus_reproduction.csv"
    nucleotides_dir = "/workspaces/src/bionemo-framework/attack/data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides"

    ref_df = pd.read_csv(ref_csv_path)
    print(f"Found {len(ref_df)} datasets in reference file")

    # Initialize HDF5 files for splits
    if getattr(args, 'save_rest', False):
        train_path = None
        test_path = None
        rest_path = os.path.join(output_dir, 'probe_rest.h5')
    else:
        train_path = os.path.join(output_dir, 'probe_train.h5')
        test_path = os.path.join(output_dir, 'probe_test.h5')
        rest_path = None
    
    metadata = {
        'n_samples_per_dataset': args.n_samples,
        'seed': args.seed,
        'columns': ['sequences', 'scores', 'DMS_dataset', 'labels', 'mutant'],
        'description': 'DMS probe dataset with nucleotide sequences and fitness scores'
    }
    
    if rest_path is not None:
        initialize_hdf5_file(rest_path, metadata)
    else:
        initialize_hdf5_file(train_path, metadata)
        initialize_hdf5_file(test_path, metadata)

    # Process each dataset
    processed_datasets = []
    
    for dataset_idx, (idx, row) in enumerate(ref_df.iterrows()):
        csv_filename = str(row['csv_filename'])
        dataset_path = os.path.join(nucleotides_dir, csv_filename)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            continue
        
        print(f"Processing dataset {dataset_idx+1}/{len(ref_df)}: {csv_filename}")
        
        # Load and sample dataset
        train_df, test_df = load_and_sample_dataset(dataset_path, n_samples=args.n_samples, seed=args.seed)
        if train_df is None or test_df is None:
            continue
        
        # Extract dataset name (without .csv extension)
        dataset_name = os.path.splitext(csv_filename)[0]
        
        # Prepare data for train split
        train_sequences = train_df['nucleotide_sequence'].tolist()
        train_scores = train_df['DMS_score'].tolist()
        train_labels = train_df['DMS_score_bin'].tolist()
        train_mutants = train_df['mutant'].tolist()
        train_dms_datasets = [dataset_name] * len(train_sequences)
        
        # Prepare data for test split
        test_sequences = test_df['nucleotide_sequence'].tolist()
        test_scores = test_df['DMS_score'].tolist()
        test_labels = test_df['DMS_score_bin'].tolist()
        test_mutants = test_df['mutant'].tolist()
        test_dms_datasets = [dataset_name] * len(test_sequences)
        
        # Append to HDF5 files
        if not getattr(args, 'save_rest', False):
            append_to_hdf5(train_path, train_sequences, train_scores, train_dms_datasets, train_labels, train_mutants)
            append_to_hdf5(test_path, test_sequences, test_scores, test_dms_datasets, test_labels, test_mutants)

        # Compute and append REST split if requested
        if rest_path is not None:
            try:
                full_df = pd.read_csv(dataset_path)
                rest_df = compute_rest_df(full_df, train_df, test_df)
                if len(rest_df) > 0:
                    rest_sequences = rest_df['nucleotide_sequence'].tolist()
                    rest_scores = rest_df['DMS_score'].tolist()
                    rest_labels = rest_df['DMS_score_bin'].tolist()
                    rest_mutants = rest_df['mutant'].tolist()
                    rest_dms_datasets = [dataset_name] * len(rest_sequences)
                    append_to_hdf5(
                        rest_path,
                        rest_sequences,
                        rest_scores,
                        rest_dms_datasets,
                        rest_labels,
                        rest_mutants,
                    )
                    rest_count = len(rest_sequences)
                else:
                    rest_count = 0
            except Exception as e:
                logger.warning(f"Failed to compute REST split for {csv_filename}: {e}")
                rest_count = 0
        
        # Track this dataset for metadata
        entry = {
            'name': dataset_name,
            'train_count': len(train_sequences),
            'test_count': len(test_sequences)
        }
        if rest_path is not None:
            entry['rest_count'] = rest_count
        processed_datasets.append(entry)

        if args.test_mode:
            break

    # Save metadata according to selected mode
    if getattr(args, 'save_rest', False):
        total_rest_sequences = sum(dataset.get('rest_count', 0) for dataset in processed_datasets)
        rest_metadata = {
            'n_samples_per_dataset': args.n_samples,
            'seed': args.seed,
            'total_rest_sequences': total_rest_sequences,
            'total_datasets': len(processed_datasets),
            'datasets': [dataset['name'] for dataset in processed_datasets],
            'columns': ['sequences', 'scores', 'DMS_dataset', 'labels', 'mutant'],
            'files': ['probe_rest.h5']
        }
        rest_metadata_path = os.path.join(output_dir, 'dataset_metadata_rest.json')
        with open(rest_metadata_path, 'w') as f:
            json.dump(rest_metadata, f, indent=2)
    else:
        total_train_sequences = sum(dataset['train_count'] for dataset in processed_datasets)
        total_test_sequences = sum(dataset['test_count'] for dataset in processed_datasets)
        overall_metadata = {
            'n_samples_per_dataset': args.n_samples,
            'seed': args.seed,
            'total_train_sequences': total_train_sequences,
            'total_test_sequences': total_test_sequences,
            'total_datasets': len(processed_datasets),
            'datasets': [dataset['name'] for dataset in processed_datasets],
            'columns': ['sequences', 'scores', 'DMS_dataset', 'labels', 'mutant'],
            'files': ['probe_train.h5', 'probe_test.h5']
        }
        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(overall_metadata, f, indent=2)

    print(f"\nDataset creation completed!")
    print(f"Files saved to: {output_dir}")
    if getattr(args, 'save_rest', False):
        print(f"  - probe_rest.h5: REST sequences")
        print(f"  - dataset_metadata_rest.json: REST metadata")
    else:
        print(f"  - probe_train.h5: {total_train_sequences} training sequences")
        print(f"  - probe_test.h5: {total_test_sequences} test sequences")
        print(f"  - probe_train.h5 and probe_test.h5")
        print(f"  - dataset_metadata.json: Overall metadata")
    print(f"Processed {len(processed_datasets)} datasets")

def main():
    parser = argparse.ArgumentParser(description="Create simple DMS probe dataset without model representations")
    
    parser.add_argument("--output_dir", type=str, 
                       default="/workspaces/src/bionemo-framework/attack/data/eval_dataset/fitness/probe_datasets",
                       help="Output directory for datasets")
    parser.add_argument("--n_samples", type=int, default=624,
                       help="Total samples per dataset (will be split into train/test)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode: process only the first dataset for quick testing")
    parser.add_argument("--save_rest", action="store_true",
                       help="Also save REST split (remaining rows) to probe_rest.h5 and metadata JSON")
    
    args = parser.parse_args()
    
    print("Starting simple DMS probe dataset creation...")
    print(f"Arguments: {vars(args)}")
    
    try:
        create_probe_datasets(args)
        print("Dataset creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
