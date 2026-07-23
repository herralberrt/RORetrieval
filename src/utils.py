"""
Utility functions for RORetrieval project.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sim = cosine_similarity([a], [b])[0][0]
    return float(sim)


def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """
    Compute n-gram overlap between two texts.
    
    Args:
        text1: First text
        text2: Second text
        n: N-gram size (default: bigrams)
    
    Returns:
        Overlap score [0, 1]
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    if len(words1) < n or len(words2) < n:
        return 0.0
    
    ngrams1 = set()
    for i in range(len(words1) - n + 1):
        ngrams1.add(' '.join(words1[i:i+n]))
    
    ngrams2 = set()
    for i in range(len(words2) - n + 1):
        ngrams2.add(' '.join(words2[i:i+n]))
    
    if len(ngrams1) == 0 or len(ngrams2) == 0:
        return 0.0
    
    overlap = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
    return overlap


def lexical_overlap(text1: str, text2: str) -> float:
    """
    Compute lexical overlap (word overlap) between two texts.
    
    Returns:
        Overlap score [0, 1]
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    common = len(words1 & words2)
    total = len(words1 | words2)
    
    return common / total if total > 0 else 0.0


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_config(env_file: str = '.env') -> Dict[str, str]:
    """Load configuration from .env file."""
    config = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('#') or '=' not in line:
                    continue
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config


def batch_generator(data: List[Any], batch_size: int):
    """Generate batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def print_stats(data: List[Dict[str, Any]], keys: List[str]) -> None:
    """Print statistics about data."""
    print(f"\n Dataset Statistics (n={len(data)})")
    print("-" * 60)
    
    for key in keys:
        values = [item.get(key) for item in data if key in item]
        if values:
            if isinstance(values[0], (int, float)):
                print(f"  {key}:")
                print(f"    - Min: {min(values):.3f}")
                print(f"    - Max: {max(values):.3f}")
                print(f"    - Mean: {np.mean(values):.3f}")
                print(f"    - Std: {np.std(values):.3f}")
            else:
                print(f"  {key}: {len(set(values))} unique values")


def validate_triplet(triplet: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate triplet structure.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    required_fields = ['query', 'positive_doc_id', 'negative_doc_ids']
    for field in required_fields:
        if field not in triplet:
            errors.append(f"Missing field: {field}")
    
    if 'negative_doc_ids' in triplet:
        if not isinstance(triplet['negative_doc_ids'], list):
            errors.append("negative_doc_ids must be a list")
        elif len(triplet['negative_doc_ids']) == 0:
            errors.append("negative_doc_ids cannot be empty")
    
    return len(errors) == 0, errors
