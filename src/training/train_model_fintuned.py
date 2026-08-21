#!/usr/bin/env python3
"""
Train a fine-tuned model on Romanian retrieval task with TripletLoss
Saves model properly for inference
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import torch

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import load_jsonl, save_jsonl, ensure_dir


def train_model_with_triplet_loss():
    """Train and save a fine-tuned model"""
    
    ensure_dir("results/trained_model")
    
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ Missing dependencies: pip install sentence-transformers torch")
        return False
    
    print("="*60)
    print("Fine-tuning Model with Triplet Loss")
    print("="*60)
    
    # Load model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Load training data
    print("\nLoading training data...")
    train_triplets = load_jsonl("results/splits/train_triplets.jsonl")
    val_triplets = load_jsonl("results/splits/val_triplets.jsonl")
    
    print(f"Training triplets: {len(train_triplets)}")
    print(f"Validation triplets: {len(val_triplets)}")
    
    # Convert to InputExample format
    print("\nPreparing training examples...")
    train_examples = []
    for triplet in train_triplets:
        anchor = triplet.get("query", "")
        positive = triplet.get("positive_content", "")
        negative = triplet.get("negative_content", "")
        
        if anchor and positive and negative:
            train_examples.append(InputExample(texts=[anchor, positive, negative]))
    
    val_examples = []
    for triplet in val_triplets:
        anchor = triplet.get("query", "")
        positive = triplet.get("positive_content", "")
        negative = triplet.get("negative_content", "")
        
        if anchor and positive and negative:
            val_examples.append(InputExample(texts=[anchor, positive, negative]))
    
    print(f"Valid training examples: {len(train_examples)}")
    print(f"Valid validation examples: {len(val_examples)}")
    
    if not train_examples:
        print("❌ No valid training examples!")
        return False
    
    # Setup DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    
    # Setup loss
    train_loss = losses.TripletLoss(model=model, triplet_margin=0.5)
    
    # Training configuration
    num_epochs = 5
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    print(f"\nTraining config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 32")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {len(train_dataloader) * num_epochs}")
    
    # Train model
    print("\nTraining...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path="results/trained_model",
        save_best_model=True,
        checkpoint_save_steps=len(train_dataloader),  # Save each epoch
        checkpoint_save_total_limit=3  # Keep last 3
    )
    
    # Save model explicitly
    model_path = Path("results/trained_model") / "final_model"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\n✅ Model saved to {model_path}")
    
    # Save training config
    config = {
        "base_model": model_name,
        "training_data": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "total_triplets": len(train_triplets) + len(val_triplets)
        },
        "training_config": {
            "epochs": num_epochs,
            "batch_size": 32,
            "triplet_margin": 0.5,
            "warmup_steps": warmup_steps,
            "optimizer": "AdamW"
        },
        "model_path": str(model_path)
    }
    
    config_path = Path("results/trained_model") / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to {config_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    
    return True


def load_and_test_model():
    """Load trained model and do quick test"""
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Missing dependencies")
        return False
    
    model_path = "results/trained_model/final_model"
    
    print(f"\nLoading trained model from {model_path}...")
    model = SentenceTransformer(model_path)
    
    # Quick test
    test_texts = [
        "Cum se gătește o ciorbă tradițională?",
        "Rețeta de ciorbă de burtă",
        "Care sunt beneficiile exercițiilor fizice?"
    ]
    
    print("\nTesting model on sample queries:")
    embeddings = model.encode(test_texts)
    
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text[:60]}...")
    
    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1:]
    )[0]
    
    print(f"\nSimilarity of first query to others:")
    print(f"  To second: {similarities[0]:.4f}")
    print(f"  To third: {similarities[1]:.4f}")
    
    return True


if __name__ == "__main__":
    # Train model
    success = train_model_with_triplet_loss()
    
    if success:
        # Test loaded model
        load_and_test_model()
