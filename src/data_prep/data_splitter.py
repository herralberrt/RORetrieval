import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys
import numpy as np

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import load_jsonl, save_jsonl, ensure_dir


class DataSplitter:
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def split_triplets(self, triplets_path: str, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      output_dir: str = "results/splits") -> Dict[str, str]:
        
        ensure_dir(output_dir)
        
        triplets = load_jsonl(triplets_path)
        n_triplets = len(triplets)
        
        indices = np.arange(n_triplets)
        np.random.shuffle(indices)
        
        train_end = int(n_triplets * train_ratio)
        val_end = train_end + int(n_triplets * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_triplets = [triplets[i] for i in train_indices]
        val_triplets = [triplets[i] for i in val_indices]
        test_triplets = [triplets[i] for i in test_indices]
        
        train_path = Path(output_dir) / "train_triplets.jsonl"
        val_path = Path(output_dir) / "val_triplets.jsonl"
        test_path = Path(output_dir) / "test_triplets.jsonl"
        
        save_jsonl(train_triplets, str(train_path))
        save_jsonl(val_triplets, str(val_path))
        save_jsonl(test_triplets, str(test_path))
        
        stats = {
            "total": n_triplets,
            "train": len(train_triplets),
            "val": len(val_triplets),
            "test": len(test_triplets),
            "train_ratio": len(train_triplets) / n_triplets,
            "val_ratio": len(val_triplets) / n_triplets,
            "test_ratio": len(test_triplets) / n_triplets
        }
        
        stats_path = Path(output_dir) / "split_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Split Statistics:")
        print(f"  Total Triplets: {stats['total']}")
        print(f"  Train: {stats['train']} ({stats['train_ratio']:.2%})")
        print(f"  Val: {stats['val']} ({stats['val_ratio']:.2%})")
        print(f"  Test: {stats['test']} ({stats['test_ratio']:.2%})")
        
        print(f"\nFiles saved to {output_dir}:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print(f"  {test_path}")
        print(f"  {stats_path}")
        
        return {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "stats": str(stats_path)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split triplets into train/val/test")
    parser.add_argument("--triplets", type=str, default="results/triplets/triplets.jsonl",
                       help="Path to triplets file")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation ratio")
    parser.add_argument("--output-dir", type=str, default="results/splits",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    splitter = DataSplitter(random_seed=args.seed)
    paths = splitter.split_triplets(
        args.triplets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )
