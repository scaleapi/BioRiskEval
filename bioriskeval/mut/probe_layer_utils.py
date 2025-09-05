#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


LAYER_RE = re.compile(r"layer=(\d+)")


def _extract_layer_from_name(name: str) -> Optional[int]:
    m = LAYER_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _collect_layers_in_dir(model_dir: Path) -> Set[int]:
    layers: Set[int] = set()
    for p in model_dir.iterdir():
        if p.is_file() and (p.suffix.lower() in (".json", ".pt")):
            layer = _extract_layer_from_name(p.name)
            if layer is not None:
                layers.add(layer)
    return layers


def load_layers_by_model(probes_root: Path) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
    """Load per-model layer sets for both selection methods (if present)."""
    val_dir = probes_root / "closed_form_val_spearman"
    tr_dir = probes_root / "closed_form_train_rmse"

    def _scan(root: Path) -> Dict[str, Set[int]]:
        mapping: Dict[str, Set[int]] = {}
        if not root.exists():
            return mapping
        for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            layers = _collect_layers_in_dir(model_dir)
            if layers:
                mapping[model_dir.name] = layers
        return mapping

    return _scan(val_dir), _scan(tr_dir)


def union_layers(val_map: Dict[str, Set[int]], tr_map: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for k, v in val_map.items():
        out.setdefault(k, set()).update(v)
    for k, v in tr_map.items():
        out.setdefault(k, set()).update(v)
    return out


def derive_model_dir_from_checkpoint(checkpoint_path: str) -> List[str]:
    """Given a checkpoint_path, produce likely model directory names used in saved_probes.

    Rules observed:
    - Finetuned checkpoints live under .../ft_checkpoints/<training_config>/.../consumed_samples=N...
      Saved probes use <training_config>_samples-N (dash) or sometimes =N (equals). We return both.
    - Original checkpoints use their leaf directory name as the model dir.
    """
    path = checkpoint_path.rstrip('/')
    parts = path.split('/')

    # Case 1: finetuned
    if 'ft_checkpoints' in parts:
        try:
            idx = parts.index('ft_checkpoints')
            training_config = parts[idx + 1]
        except Exception:
            training_config = parts[-1]

        # Find consumed_samples in last filename-like segment
        last = parts[-1]
        m = re.search(r"consumed_samples=([0-9.]+)", last)
        samples = None
        if m:
            try:
                samples = int(float(m.group(1)))
            except Exception:
                samples = None

        candidates: List[str] = [training_config]
        if samples is not None:
            candidates.insert(0, f"{training_config}_samples-{samples}")
            candidates.insert(1, f"{training_config}_samples={samples}")
        return candidates

    # Case 2: original
    return [parts[-1] if parts else "unknown_model"]


def get_required_layers_for_checkpoint(probes_root: Path, checkpoint_path: str) -> List[int]:
    val_map, tr_map = load_layers_by_model(probes_root)
    uni = union_layers(val_map, tr_map)
    candidates = derive_model_dir_from_checkpoint(checkpoint_path)

    # Exact match first
    for cand in candidates:
        if cand in uni:
            return sorted(list(uni[cand]))

    # Fuzzy: startswith training_config and contains sample number
    if len(candidates) >= 1:
        base = candidates[-1]
        sample_nums = [s for s in candidates if 'samples' in s]
        sample_str = None
        if sample_nums:
            # pick digits after samples[-=]
            m = re.search(r"samples[-=]([0-9]+)", sample_nums[0])
            if m:
                sample_str = m.group(1)
        for k in uni.keys():
            if k.startswith(base) and (sample_str is None or (sample_str in k)):
                return sorted(list(uni[k]))

    return []


def format_decoder_layer_names(layers: List[int]) -> List[str]:
    return [f"decoder.layers.{n}" for n in sorted(set(layers))]


def main():
    parser = argparse.ArgumentParser(description="Compute required probe layers per checkpoint from saved_probes")
    parser.add_argument("--probes_root", type=str, required=True, help="Root of saved_probes directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint path to resolve model dir and layers")
    parser.add_argument("--format", type=str, default="json", choices=["json", "bash"], help="Output format: json (default) or bash (space-separated)")
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    layers = get_required_layers_for_checkpoint(probes_root, args.checkpoint_path)
    layer_names = format_decoder_layer_names(layers)

    if args.format == "bash":
        print(" ".join(layer_names))
    else:
        print(json.dumps({
            "checkpoint_path": args.checkpoint_path,
            "probes_root": str(probes_root),
            "layers": layers,
            "layer_names": layer_names,
        }, indent=2))


if __name__ == "__main__":
    main()


