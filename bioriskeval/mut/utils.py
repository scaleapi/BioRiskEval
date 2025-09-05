import re

def extract_model_name(checkpoint_path):
    """Extract meaningful model name from checkpoint path."""
    path = checkpoint_path.rstrip('/')

    # Check if this is a fine-tuned model path
    if 'ft_checkpoints' in path:
        parts = path.split('/')

        # Find the training configuration directory (after ft_checkpoints)
        training_config = None
        for i, part in enumerate(parts):
            if 'ft_checkpoints' in part and i + 1 < len(parts):
                training_config = parts[i + 1]
                break

        if not training_config:
            # Fallback if structure is unexpected
            return parts[-2] if len(parts) > 1 else parts[-1]

        # Look for consumed_samples in the checkpoint filename (last part)
        checkpoint_file = parts[-1]
        samples_match = re.search(r'consumed_samples=([0-9.]+)', checkpoint_file)

        if samples_match:
            samples_value = int(float(samples_match.group(1)))
            return f"{training_config}_samples={samples_value}"
        else:
            return training_config

    # Handle original model paths
    else:
        parts = path.split('/')
        return parts[-1] if parts else "unknown_model" 