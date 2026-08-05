import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, ensure_dir


class FaissIndexer:
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 index_dir: str = "results/faiss"):
        self.model_name = model_name
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.id_to_docid = {}
        self.doc_count = 0
        ensure_dir(index_dir)
    
    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def embed_documents(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        texts = []
        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            text = f"{title} {content}".strip()
            texts.append(text)
        
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings
    
    def build_index(self, corpus_path: str):
        if not self.load_model():
            return False
        
        print(f"Loading corpus from {corpus_path}")
        documents = load_jsonl(corpus_path)
        
        print(f"Embedding {len(documents)} documents")
        embeddings = self.embed_documents(documents)
        
        import faiss
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        for idx, doc in enumerate(documents):
            self.id_to_docid[idx] = doc.get("doc_id", f"doc_{idx}")
        
        self.doc_count = len(documents)
        self.save_index()
        
        print(f"Index built with {self.doc_count} documents")
        return True
    
    def save_index(self):
        import faiss
        index_path = os.path.join(self.index_dir, "corpus.index")
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")
        
        id_mapping_path = os.path.join(self.index_dir, "id_mapping.json")
        with open(id_mapping_path, 'w') as f:
            json.dump(self.id_to_docid, f)
        print(f"ID mapping saved to {id_mapping_path}")
    
    def load_index(self):
        import faiss
        index_path = os.path.join(self.index_dir, "corpus.index")
        id_mapping_path = os.path.join(self.index_dir, "id_mapping.json")
        
        if not os.path.exists(index_path):
            print(f"Index not found at {index_path}")
            return False
        
        self.index = faiss.read_index(index_path)
        print(f"Index loaded from {index_path}")
        
        with open(id_mapping_path, 'r') as f:
            mapping = json.load(f)
            self.id_to_docid = {int(k): v for k, v in mapping.items()}
        
        self.doc_count = len(self.id_to_docid)
        print(f"Loaded {self.doc_count} documents from index")
        return True
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model()
        
        query_embedding = self.model.encode([query_text])[0]
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            top_k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "rank": len(results) + 1,
                "doc_id": self.id_to_docid[idx],
                "index_id": int(idx),
                "distance": float(distances[0][i])
            })
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        if self.model is None:
            self.load_model()
        
        query_embeddings = self.model.encode(queries, show_progress_bar=True)
        distances, indices = self.index.search(
            query_embeddings.astype('float32'), 
            top_k
        )
        
        all_results = []
        for i in range(len(queries)):
            results = []
            for rank, idx in enumerate(indices[i]):
                if idx == -1:
                    continue
                results.append({
                    "rank": rank + 1,
                    "doc_id": self.id_to_docid[idx],
                    "index_id": int(idx),
                    "distance": float(distances[i][rank])
                })
            all_results.append(results)
        
        return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build or load Faiss index")
    parser.add_argument("--corpus", type=str, default="data/corpus/all_documents_combined.jsonl", help="Corpus path")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model name")
    parser.add_argument("--index-dir", type=str, default="results/faiss", help="Index directory")
    parser.add_argument("--build", action="store_true", help="Build index")
    parser.add_argument("--query", type=str, help="Test query")
    
    args = parser.parse_args()
    
    indexer = FaissIndexer(model_name=args.model, index_dir=args.index_dir)
    
    if args.build:
        indexer.build_index(args.corpus)
    elif args.query:
        if indexer.load_index():
            results = indexer.search(args.query, top_k=10)
            for result in results:
                print(f"Rank {result['rank']}: {result['doc_id']} (distance: {result['distance']:.4f})")
    else:
        if indexer.load_index():
            print(f"Index ready with {indexer.doc_count} documents")
