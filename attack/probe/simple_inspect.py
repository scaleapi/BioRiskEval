#!/usr/bin/env python3
"""
Simple model structure inspector for BioNeMo models.
CUDA_VISIBLE_DEVICES=4,5,6,7 python probe/simple_inspect.py \
    /workspaces/BioRiskEval/attack/checkpoints/orig_checkpoints/nemo2_evo2_7b_1m \
        7b_arc_longcontext | grep -A 400 "Hookable"
"""

import torch
import nemo.lightning as nl
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# Import from eval_ppl.py like eval_fitness.py does
import sys
import os
# Add the eval directory to the path (relative to ft-attack directory)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
from eval_ppl import HyenaPredictor, hyena_predict_forward_step, hyena_predict_data_step

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

def simple_inspect(checkpoint_path, model_size):
    print(f"Loading model: {model_size}")
    
    # Load tokenizer first
    tokenizer = get_nmt_tokenizer("byte-level")
    print(f"Tokenizer loaded: {type(tokenizer)}")
    
    # Create model config instance like eval_fitness.py
    print(f"Creating config instance for: {model_size}")
    config = HYENA_MODEL_OPTIONS[model_size](
        forward_step_fn=hyena_predict_forward_step,
        data_step_fn=hyena_predict_data_step,
        distribute_saved_activations=True,  # Simple case, no sequence parallel
    )
    print(f"Config instance created: {type(config)}")
    
    # Create model using official BioNeMo class
    model = HyenaPredictor(config, tokenizer=tokenizer)
    print(f"Model created: {type(model)}")
    
    # Create trainer
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
    
    # Load checkpoint using eval_fitness.py pattern
    print(f"Loading checkpoint from: {checkpoint_path}")
    resume = nl.AutoResume(
        resume_if_exists=False,  # Match eval_fitness.py
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        restore_config=nl.RestoreConfig(
            path=str(checkpoint_path),
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=True,  # Match eval_fitness.py
        ),
    )
    
    # Disable optimizer setup like eval_fitness.py
    trainer.strategy._setup_optimizers = False  # type: ignore
    
    # Setup model with checkpoint
    print("Setting up checkpoint loading...")
    resume.setup(trainer, model)
    print("Checkpoint setup completed!")
    
    # Create simple data module for inspection
    datamodule = SimpleInspectionDataModule(tokenizer, batch_size=1)
    
    # Run a simple prediction like eval_fitness.py does - this properly initializes the model
    print("Running test prediction to initialize model...")
    try:
        results = trainer.predict(model, datamodule=datamodule)
        print(f"Test prediction completed, got {len(results)} batches of results")
        
        # Now check the model parameters after it's been properly initialized
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully with {param_count:,} parameters")
        
    except Exception as e:
        print(f"Error during test prediction: {e}")
        # Fall back to direct inspection
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Direct parameter count: {param_count:,}")
    
    # Now inspect the model structure
    print("\n=== Model Structure Inspection ===")
    
    # List top-level modules
    print("\n=== Top-Level Modules ===")
    top_modules = list(model.named_children())
    if not top_modules:
        print("No top-level modules found!")
        
        # Try to find the actual model in common wrapper attributes
        print("Checking for wrapped model...")
        for attr in ['model', 'module', '_model', 'net', 'transformer', 'backbone']:
            if hasattr(model, attr):
                sub_model = getattr(model, attr)
                print(f"Found {attr}: {type(sub_model)}")
                sub_modules = list(sub_model.named_children())
                if sub_modules:
                    print(f"  Modules in {attr}:")
                    for name, mod in sub_modules:  # Show all modules
                        param_count = sum(p.numel() for p in mod.parameters())
                        print(f"    {name}: {type(mod).__name__} ({param_count:,} params)")
    else:
        for name, module in top_modules:
            module_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {type(module).__name__} ({module_params:,} params)")
    
    # Handle DDP/Float16Module wrappers - unwrap to find the actual model
    actual_model = model
    unwrapping_path = []
    
    # Common wrapper patterns in distributed training
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
    
    # Now inspect the actual model
    print(f"\n=== Actual Model Structure ===")
    actual_modules = list(actual_model.named_children())
    if actual_modules:
        print("Main model components:")
        for name, module in actual_modules:
            module_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {type(module).__name__} ({module_params:,} params)")
            
            # For decoder/transformer components, look one level deeper
            if 'decoder' in name.lower() or 'transformer' in name.lower() or 'layers' in name.lower():
                sub_modules = list(module.named_children())
                print(f"    └─ Contains {len(sub_modules)} submodules:")
                for subname, submod in sub_modules:  # Show all submodules
                    submod_params = sum(p.numel() for p in submod.parameters())
                    print(f"      {subname}: {type(submod).__name__} ({submod_params:,} params)")
    
    # Try to go deeper if we have modules
    if actual_modules:
        print("\n=== Second-Level Modules (Detailed) ===")
        for name, module in actual_modules:  # Show all modules
            print(f"\n{name} ({type(module).__name__}):")
            sub_modules = list(module.named_children())
            if not sub_modules:
                print(f"  No submodules in {name}")
            else:
                for subname, submodule in sub_modules:  # Show all submodules
                    submodule_params = sum(p.numel() for p in submodule.parameters())
                    print(f"  {subname}: {type(submodule).__name__} ({submodule_params:,} params)")
                    
                    # For layer containers, look even deeper
                    if any(keyword in subname.lower() for keyword in ['layers', 'blocks']):
                        third_level = list(submodule.named_children())
                        if third_level:
                            print(f"    └─ Individual layers:")
                            for i, (layer_name, layer_mod) in enumerate(third_level):  # Show all layers
                                layer_params = sum(p.numel() for p in layer_mod.parameters())
                                print(f"      {layer_name}: {type(layer_mod).__name__} ({layer_params:,} params)")
    
    # Look for hookable layers in the unwrapped model
    print("\n=== Looking for Hookable Layers ===")
    def find_hookable_layers(module, prefix="", max_depth=4):
        hookable = []
        if max_depth <= 0:
            return hookable
            
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Look for common patterns we can hook
            if any(keyword in name.lower() for keyword in ['mlp', 'attn', 'attention', 'mixer', 'layer', 'block']):
                child_params = sum(p.numel() for p in child.parameters())
                if child_params > 0:  # Only include layers with parameters
                    hookable.append((full_name, type(child).__name__, child_params))
                    print(f"Found hookable: {full_name} ({type(child).__name__}, {child_params:,} params)")
                    
                    # Look one level deeper for sub-components
                    for subname, subchild in child.named_children():
                        sub_full_name = f"{full_name}.{subname}"
                        if any(kw in subname.lower() for kw in ['mlp', 'attn', 'mixer', 'linear', 'conv', 'norm']):
                            sub_params = sum(p.numel() for p in subchild.parameters())
                            if sub_params > 0:
                                print(f"  -> {sub_full_name}: {type(subchild).__name__} ({sub_params:,} params)")
            
            # Also look for layer containers (numbered layers)
            if name.isdigit() or ('layer' in name.lower() and any(c.isdigit() for c in name)):
                child_params = sum(p.numel() for p in child.parameters())
                if child_params > 0:
                    print(f"Found layer container: {full_name} ({type(child).__name__}, {child_params:,} params)")
                    # Look inside layer containers
                    for subname, subchild in child.named_children():
                        sub_full_name = f"{full_name}.{subname}"
                        sub_params = sum(p.numel() for p in subchild.parameters())
                        if sub_params > 0:
                            hookable.append((sub_full_name, type(subchild).__name__, sub_params))
                            print(f"  -> {sub_full_name}: {type(subchild).__name__} ({sub_params:,} params)")
            
            # Recurse
            hookable.extend(find_hookable_layers(child, full_name, max_depth - 1))
        
        return hookable
    
    hookable_layers = find_hookable_layers(actual_model)
    
    if hookable_layers:
        print(f"\nFound {len(hookable_layers)} hookable layers!")
        # print("Suggested layer names for hook testing:")
        # for layer_name, layer_type, param_count in hookable_layers:  # Show all hookable layers
        #     print(f"  - {layer_name}")
    else:
        print("\n❌ No obvious hookable layers found")
        print("Let's try a more aggressive search...")
        
        # More aggressive search - look for any module with substantial parameters
        def aggressive_search(module, prefix="", max_depth=5):
            results = []
            if max_depth <= 0:
                return results
                
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                child_params = sum(p.numel() for p in child.parameters())
                
                # Include any module with significant parameters
                if child_params > 1000:  # At least 1k parameters
                    results.append((full_name, type(child).__name__, child_params))
                
                # Recurse
                results.extend(aggressive_search(child, full_name, max_depth - 1))
            
            return results
        
        all_layers = aggressive_search(actual_model)
        if all_layers:
            print(f"Found {len(all_layers)} modules with parameters:")
            for layer_name, layer_type, param_count in all_layers:  # Show all layers
                print(f"  - {layer_name}: {layer_type} ({param_count:,} params)")
    
    return actual_model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simple_inspect.py <checkpoint_path> <model_size>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    model_size = sys.argv[2]
    
    model = simple_inspect(checkpoint_path, model_size) 