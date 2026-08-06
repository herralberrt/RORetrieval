import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir


class SentenceTransformerTrainer:
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 corpus_path: str = "data/corpus/all_documents_combined.jsonl"):
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.corpus = load_jsonl(corpus_path)
        self.corpus_by_id = {doc["doc_id"]: doc for doc in self.corpus}
        self.model = None
    
    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def prepare_triplet_examples(self, triplets: List[Dict[str, Any]]):
        from sentence_transformers import InputExample
        
        examples = []
        for triplet in triplets:
            anchor = triplet.get("query", "")
            positive = triplet.get("positive_content", "")
            negative = triplet.get("negative_content", "")
            
            if anchor and positive and negative:
                examples.append(InputExample(texts=[anchor, positive, negative]))
        
        return examples
    
    def train(self, train_triplets_path: str, val_triplets_path: str = None,
             epochs: int = 3, batch_size: int = 32, warmup_steps: int = 100,
             output_dir: str = "results/trained_model_v2"):
        
        ensure_dir(output_dir)
        
        train_triplets = load_jsonl(train_triplets_path)
        val_triplets = load_jsonl(val_triplets_path) if val_triplets_path else []
        
        if not self.load_model():
            return False
        
        print(f"Preparing {len(train_triplets)} training triplets...")
        train_examples = self.prepare_triplet_examples(train_triplets)
        
        print(f"Preparing {len(val_triplets)} validation triplets...")
        val_examples = self.prepare_triplet_examples(val_triplets) if val_triplets else []
        
        from torch.utils.data import DataLoader
        from sentence_transformers import losses
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        train_loss = losses.TripletLoss(model=self.model, triplet_margin=0.5)
        
        print(f"\nStarting fine-tuning...")
        print(f"  Model: {self.model_name}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Output directory: {output_dir}\n")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir,
            show_progress_bar=True,
            checkpoint_save_steps=len(train_dataloader) // 2,
            checkpoint_save_total_limit=2
        )
        
        print(f"\nFine-tuning completed!")
        print(f"Model saved to {output_dir}")
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer on triplets")
    parser.add_argument("--corpus", type=str, default="data/corpus/all_documents_combined.jsonl",
                       help="Corpus path")
    parser.add_argument("--train-triplets", type=str, default="results/splits/train_triplets.jsonl",
                       help="Training triplets path")
    parser.add_argument("--val-triplets", type=str, default="results/splits/val_triplets.jsonl",
                       help="Validation triplets path")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--output-dir", type=str, default="results/trained_model_v2",
                       help="Output directory")
    parser.add_argument("--model", type=str, 
                       default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       help="Model name")
    
    args = parser.parse_args()
    
    trainer = SentenceTransformerTrainer(model_name=args.model, corpus_path=args.corpus)
    trainer.train(
        train_triplets_path=args.train_triplets,
        val_triplets_path=args.val_triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir
    )
