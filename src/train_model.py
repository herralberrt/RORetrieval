import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir


class TripletLoss:
    
    def __init__(self, margin: float = 0.5):
        self.margin = margin
    
    def compute_loss(self, anchor_emb: np.ndarray, pos_emb: np.ndarray, 
                    neg_emb: np.ndarray) -> float:
        pos_distance = np.linalg.norm(anchor_emb - pos_emb)
        neg_distance = np.linalg.norm(anchor_emb - neg_emb)
        
        loss = max(0, self.margin + pos_distance - neg_distance)
        return loss


class ModelTrainer:
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 corpus_path: str = "data/corpus/all_documents_combined.jsonl"):
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.corpus = load_jsonl(corpus_path)
        self.corpus_by_id = {doc["doc_id"]: doc for doc in self.corpus}
        self.model = None
        self.loss_fn = TripletLoss(margin=0.5)
        self.training_history = []
    
    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
    
    def compute_batch_loss(self, triplets: List[Dict[str, Any]]) -> float:
        total_loss = 0.0
        
        for triplet in triplets:
            query = triplet.get("query", "")
            pos_content = triplet.get("positive_content", "")
            neg_content = triplet.get("negative_content", "")
            
            anchor_emb = self.encode_text(query)
            pos_emb = self.encode_text(pos_content)
            neg_emb = self.encode_text(neg_content)
            
            loss = self.loss_fn.compute_loss(anchor_emb, pos_emb, neg_emb)
            total_loss += loss
        
        return total_loss / len(triplets) if triplets else 0.0
    
    def train(self, train_triplets_path: str, val_triplets_path: str = None,
             epochs: int = 3, batch_size: int = 32,
             output_dir: str = "results/trained_model"):
        
        ensure_dir(output_dir)
        
        train_triplets = load_jsonl(train_triplets_path)
        val_triplets = load_jsonl(val_triplets_path) if val_triplets_path else []
        
        if not self.load_model():
            return False
        
        print(f"Training on {len(train_triplets)} triplets")
        print(f"Validation on {len(val_triplets)} triplets")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in tqdm(range(0, len(train_triplets), batch_size), 
                         desc=f"Epoch {epoch+1}/{epochs}"):
                batch = train_triplets[i:i+batch_size]
                batch_loss = self.compute_batch_loss(batch)
                epoch_loss += batch_loss * len(batch)
            
            epoch_loss /= len(train_triplets)
            
            val_loss = 0.0
            if val_triplets:
                for i in range(0, len(val_triplets), batch_size):
                    batch = val_triplets[i:i+batch_size]
                    batch_loss = self.compute_batch_loss(batch)
                    val_loss += batch_loss * len(batch)
                val_loss /= len(val_triplets)
            
            history_entry = {
                "epoch": epoch + 1,
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss) if val_triplets else None
            }
            
            self.training_history.append(history_entry)
            
            if val_triplets:
                print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}")
        
        history_path = Path(output_dir) / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Training history saved to {history_path}")
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model on triplets")
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
    parser.add_argument("--output-dir", type=str, default="results/trained_model",
                       help="Output directory")
    parser.add_argument("--model", type=str, 
                       default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       help="Model name")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(model_name=args.model, corpus_path=args.corpus)
    trainer.train(
        train_triplets_path=args.train_triplets,
        val_triplets_path=args.val_triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
