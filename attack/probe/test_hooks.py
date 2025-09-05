#!/usr/bin/env python3
"""
Test script to validate hook functionality in BioNeMo Evo2 models.

Usage:
CUDA_VISIBLE_DEVICES=4,5,6,7 python probe/test_hooks.py \
    --checkpoint_path /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m \
    --layer_name "decoder.layers.26" \
    --model_size 7b_arc_longcontext 
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import nemo.lightning as nl
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from megatron.core import parallel_state

# Import from eval_ppl.py like eval_fitness.py does
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
from eval_ppl import HyenaPredictor, hyena_predict_forward_step, hyena_predict_data_step

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphSafeFeatureExtractor:
    """Feature extractor that works with torch.compile and CUDA graphs."""
    
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = {}
        
    def _create_graph_safe_hook(self, layer_name):
        """Create a hook that immediately detaches outputs from the computation graph."""
        def hook_fn(module, input, output):
            # Immediately detach and clone to break graph dependencies
            if isinstance(output, torch.Tensor):
                # Clone and detach to completely separate from computation graph
                safe_output = output.detach().clone()
            elif isinstance(output, (tuple, list)):
                # Handle multiple outputs
                safe_output = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output)
            else:
                safe_output = output
                
            # Store in a way that doesn't interfere with the main computation
            self.features[layer_name] = safe_output
            
        return hook_fn
    
    def register_hooks(self):
        """Register graph-safe hooks on target layers."""
        # Find the actual model layers using robust unwrapping like simple_inspect.py
        actual_model = self.model
        unwrapping_path = []
        
        # Common wrapper patterns in distributed training (from simple_inspect.py)
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
        
        # Navigate to the decoder layers
        if hasattr(actual_model, 'decoder'):
            decoder = actual_model.decoder
            logger.info(f"Found decoder: {type(decoder).__name__}")
        else:
            # Try alternative paths
            for attr_name in ['transformer', 'layers', 'blocks']:
                if hasattr(actual_model, attr_name):
                    decoder = getattr(actual_model, attr_name)
                    logger.info(f"Found decoder via {attr_name}: {type(decoder).__name__}")
                    break
            else:
                # List available attributes for debugging
                attrs = [attr for attr in dir(actual_model) if not attr.startswith('_')]
                logger.error(f"Cannot find decoder/transformer layers in {type(actual_model)}")
                logger.error(f"Available attributes: {attrs[:10]}...")  # Show first 10
                raise AttributeError(f"Cannot find decoder/transformer layers in {type(actual_model)}")
        
        # Register hooks on target layers
        for layer_name in self.layer_names:
            logger.info(f"Attempting to register hook on: {layer_name}")
            
            # Parse layer path (e.g., "decoder.layers.15.mlp")
            parts = layer_name.split('.')
            
            # Start from decoder
            current = decoder
            
            # Navigate to the target layer (skip 'decoder' since we already have it)
            path_parts = parts[1:] if parts[0] == 'decoder' else parts
            
            for i, part in enumerate(path_parts):
                try:
                    if part.isdigit():
                        current = current[int(part)]
                        logger.info(f"  Navigated to index {part}: {type(current).__name__}")
                    else:
                        current = getattr(current, part)
                        logger.info(f"  Navigated to attribute {part}: {type(current).__name__}")
                except (AttributeError, IndexError, TypeError) as e:
                    logger.error(f"  Failed to navigate to {part} at step {i}: {e}")
                    logger.error(f"  Current object: {type(current).__name__}")
                    if hasattr(current, '__dict__'):
                        attrs = [attr for attr in dir(current) if not attr.startswith('_')]
                        logger.error(f"  Available attributes: {attrs[:10]}...")
                    raise
            
            # Register the graph-safe hook
            hook = current.register_forward_hook(self._create_graph_safe_hook(layer_name))
            self.hooks[layer_name] = hook
            logger.info(f"✅ Registered graph-safe hook on {layer_name}")
    

    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.features.clear()


# Use the same data classes as simple_inspect.py
class SimpleInspectionDataset(Dataset):
    """Simple dataset for model inspection."""
    
    def __init__(self, tokenizer, num_samples=2):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        # Create simple test sequences
        self.sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"][:num_samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.text_to_ids(sequence)
        
        # Ensure minimum length
        if len(tokens) < 10:
            tokens = tokens + [self.tokenizer.pad_id] * (10 - len(tokens))
        
        tokens = torch.tensor(tokens[:50], dtype=torch.long)  # Limit length
        position_ids = torch.arange(len(tokens), dtype=torch.long)
        loss_mask = torch.ones(len(tokens), dtype=torch.long)
        
        return {
            "tokens": tokens,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "seq_idx": torch.tensor(idx, dtype=torch.long),
        }

class SimpleInspectionDataModule(LightningDataModule):
    """Simple data module for model inspection."""
    
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
        """Simple collate function."""
        # Find max length
        max_len = max(len(item["tokens"]) for item in batch)
        
        # Pad sequences
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


def test_graph_safe_extraction(model, layer_names, tokenizer, trainer):
    """Test graph-safe feature extraction using Lightning predict."""
    
    # Check if we're on rank 0 for distributed training
    try:
        tensor_rank = parallel_state.get_tensor_model_parallel_rank()
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        data_rank = parallel_state.get_data_parallel_rank()
        
        is_rank_zero = tensor_rank == 0 and pipeline_rank == 0 and data_rank == 0
        
        if not is_rank_zero:
            logger.info(f"Not on rank 0 (tensor_rank={tensor_rank}, "
                       f"pipeline_rank={pipeline_rank}, data_rank={data_rank}), skipping feature extraction")
            return True  # Non-rank-0 processes should succeed without doing work
    except Exception as e:
        logger.warning(f"Could not get parallel state ranks: {e}, proceeding anyway")
        # If parallel state is not initialized, assume single GPU and proceed
    
    logger.info("Running on rank 0, proceeding with feature extraction")
    
    # Create feature extractor
    extractor = GraphSafeFeatureExtractor(model, layer_names)
    
    try:
        # Register hooks
        extractor.register_hooks()
        
        # Clear any previous features
        extractor.features.clear()
        
        # Create a simple data module for feature extraction
        datamodule = SimpleInspectionDataModule(tokenizer, batch_size=1)
        
        # Extract features using Lightning's predict method
        print("Running graph-safe feature extraction via Lightning predict...")
        
        # Run prediction - this will trigger our hooks
        results = trainer.predict(model, datamodule=datamodule)
        
        # Get the features that were captured by our hooks
        features = extractor.features.copy()
        
        # Report results
        if not features:
            print("⚠️ No features were captured by hooks")
            return False
            
        for layer_name, feature_tensor in features.items():
            if isinstance(feature_tensor, torch.Tensor):
                print(f"✅ {layer_name}: extracted features with shape {feature_tensor.shape}")
            elif isinstance(feature_tensor, (tuple, list)):
                print(f"✅ {layer_name}: extracted features tuple/list with {len(feature_tensor)} elements")
                if len(feature_tensor) > 0 and isinstance(feature_tensor[0], torch.Tensor):
                    print(f"    First element shape: {feature_tensor[0].shape}")
            else:
                print(f"✅ {layer_name}: extracted features of type {type(feature_tensor)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Test graph-safe feature extraction")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to BioNeMo Evo2 model checkpoint")
    parser.add_argument("--model_size", type=str, default="7b_arc_longcontext", 
                       choices=sorted(HYENA_MODEL_OPTIONS.keys()),
                       help="Model size configuration")
    parser.add_argument("--layer_name", type=str, default="decoder.layers.15.mlp",
                       help="Layer name to extract features from")
    
    args = parser.parse_args()
    layer_names = [args.layer_name]
    
    try:
        print("=== Starting test_hooks.py ===")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = get_nmt_tokenizer("byte-level")
        print(f"Tokenizer loaded: {type(tokenizer)}")
        
        # Create model config instance like simple_inspect.py
        print(f"Creating config instance for: {args.model_size}")
        config = HYENA_MODEL_OPTIONS[args.model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            distribute_saved_activations=True,  # Simple case, no sequence parallel
        )
        print(f"Config instance created: {type(config)}")
        
        # Create model using official BioNeMo class
        print("Creating HyenaPredictor model...")
        model = HyenaPredictor(config, tokenizer=tokenizer)
        print(f"Model created: {type(model)}")
        print("About to create trainer...")
        
        # Create trainer using exact same config as simple_inspect.py
        print("Creating Lightning trainer...")
        trainer = nl.Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy=nl.MegatronStrategy(
                drop_last_batch=False,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                ckpt_load_optimizer=False,
                ckpt_save_optimizer=False,
                ckpt_async_save=False,
                sequence_parallel=False,
                ckpt_load_strictness=None,
                data_sampler=nl.MegatronDataSampler(
                    micro_batch_size=1,
                    global_batch_size=1,
                    seq_len=8192,
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
        
        logger.info("Trainer created successfully!")
        
        # Load checkpoint using exact same pattern as simple_inspect.py
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        resume = nl.AutoResume(
            resume_if_exists=False,  # Match simple_inspect.py
            resume_ignore_no_checkpoint=False,
            resume_past_end=False,
            restore_config=nl.RestoreConfig(
                path=str(args.checkpoint_path),
                load_model_state=True,
                load_optim_state=False,
                load_artifacts=True,  # Match simple_inspect.py
            ),
        )
        
        # Disable optimizer setup like simple_inspect.py
        trainer.strategy._setup_optimizers = False  # type: ignore
        
        # Setup model with checkpoint
        logger.info("Setting up checkpoint loading...")
        resume.setup(trainer, model)
        logger.info("Checkpoint setup completed!")
        
        # Create simple data module for initialization
        datamodule = SimpleInspectionDataModule(tokenizer, batch_size=1)
        
        # Run a simple prediction like simple_inspect.py does - this properly initializes the model
        logger.info("Running test prediction to initialize model...")
        try:
            results = trainer.predict(model, datamodule=datamodule)
            logger.info(f"Test prediction completed, got {len(results)} batches of results")
            
            # Now check the model parameters after it's been properly initialized
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Model loaded successfully with {param_count:,} parameters")
            
        except Exception as e:
            logger.warning(f"Test prediction failed: {e}, continuing anyway...")
            # Fall back to direct inspection
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Direct parameter count: {param_count:,}")
        
        # Test graph-safe feature extraction
        logger.info(f"Starting feature extraction test for layer: {args.layer_name}")
        
        # Log rank information (use eval_fitness.py approach)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            current_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            print(f"Distributed training detected - rank: {current_rank}, world_size: {world_size}")
        else:
            print("Single process execution (no distributed training)")
        
        success = test_graph_safe_extraction(model, layer_names, tokenizer, trainer)
        
        if success:
            logger.info("🎉 Graph-safe feature extraction working correctly!")
            logger.info("=== test_hooks.py completed successfully ===")
            return 0
        else:
            logger.error("❌ Feature extraction failed")
            logger.info("=== test_hooks.py completed with failure ===")
            return 1
            
    except Exception as e:
        logger.error(f"Error in main(): {e}")
        import traceback
        traceback.print_exc()
        logger.error("=== test_hooks.py crashed ===")
        return 1


if __name__ == "__main__":
    print("=== test_hooks.py script starting ===")
    exit_code = main()
    print(f"=== test_hooks.py script ending with exit code {exit_code} ===")
    exit(exit_code) 