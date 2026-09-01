"""
ColBERT: Contextualized Late Interaction over BERT.

Late interaction retrieval: documents and queries are encoded as sequences of
embeddings (not a single vector like dense), and scoring is done token-by-token
using MaxSim: for each query token embedding, find the max similarity to any
document token embedding, then sum.

Why ColBERT > BM25:
- Token-level semantic understanding (not just lexical matching)
- Efficient: no need to embed the full corpus at query time
- Works great for multilingual (Romanian in this case)
- Better than dense for retrieval (dense throws away positional info)

Paper: https://arxiv.org/abs/2004.12832
"""

import sys
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from utils import load_jsonl


class ColBERTIndex:
    """ColBERT index: token embeddings for documents and queries."""
    
    def __init__(self, documents: Sequence[Dict],
                 model_name: str = "colbert-ir/colbertv2.0",
                 device: str = "cuda"):
        """
        Args:
            documents: Corpus documents (with 'title' and 'content')
            model_name: ColBERT model name (from HF or local)
            device: "cuda" or "cpu"
        """
        self.documents = list(documents)
        self.model_name = model_name
        self.device = device
        
        self.colbert = None
        self.doc_embeddings = None  # List of (doc_id, token_embeddings)
    
    def load_model(self):
        """Load ColBERT model from HuggingFace."""
        try:
            from colbert.infra import ColBERTConfig
            from colbert.modeling.colbert import ColBERT as ColBERTModel
        except ImportError:
            raise ImportError(
                "Install ColBERT: pip install colbert-ir\n"
                "Or use: pip install git+https://github.com/stanford-futuredata/ColBERT.git"
            )
        
        print(f"Loading ColBERT model: {self.model_name}")
        config = ColBERTConfig(
            doc_maxlen=220,
            query_maxlen=32,
            checkpoint=self.model_name
        )
        self.colbert = ColBERTModel.from_pretrained(config.checkpoint,
                                                      config=config)
        self.colbert = self.colbert.to(self.device)
        self.colbert.eval()
    
    def build_index(self):
        """Build embeddings for all documents."""
        if self.colbert is None:
            self.load_model()
        
        print(f"Encoding {len(self.documents)} documents with ColBERT...")
        
        self.doc_embeddings = []
        
        with torch.no_grad():
            for i, doc in enumerate(self.documents):
                if (i + 1) % 1000 == 0:
                    print(f"  Encoded {i+1}/{len(self.documents)}")
                
                doc_id = doc.get("doc_id", f"doc_{i}")
                title = doc.get("title", "")
                content = doc.get("content", "")
                text = f"{title} {content}".strip()
                
                # Encode document: returns embeddings of shape (seq_len, hidden_dim)
                D = self.colbert.docFromText([text], bsize=1)  # [1, seq_len, 128]
                self.doc_embeddings.append((doc_id, D[0].cpu()))  # Store on CPU
        
        print(f"✓ Built index for {len(self.doc_embeddings)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using ColBERT scoring (MaxSim)."""
        if self.colbert is None or self.doc_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        with torch.no_grad():
            # Encode query: returns embeddings of shape (1, seq_len, hidden_dim)
            Q = self.colbert.queryFromText([query], bsize=1)  # [1, seq_len, 128]
            Q = Q[0]  # Remove batch dimension: [seq_len, 128]
        
        # Score all documents
        scores = []
        for doc_id, D in self.doc_embeddings:
            # MaxSim: for each query token, find max similarity to document tokens
            # D shape: [doc_seq_len, 128]
            # Q shape: [query_seq_len, 128]
            
            # Cosine similarity matrix: [query_seq_len, doc_seq_len]
            similarity = torch.nn.functional.cosine_similarity(
                Q.unsqueeze(1),  # [query_seq_len, 1, 128]
                D.unsqueeze(0),  # [1, doc_seq_len, 128]
                dim=2
            )
            
            # MaxSim: max similarity for each query token, then sum
            max_sims = similarity.max(dim=1)[0]  # [query_seq_len]
            score = max_sims.sum().item()
            
            scores.append((doc_id, score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# Optional: simpler version without full ColBERT, using just sentence-transformers
# for token embeddings (faster, less memory)
class ColBERTLiteIndex:
    """Simplified ColBERT using sentence-transformers for embeddings."""
    
    def __init__(self, documents: Sequence[Dict],
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            documents: Corpus documents
            model_name: Sentence transformer model (tokenizes into subword tokens)
        """
        self.documents = list(documents)
        self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None
    
    def load_model(self):
        """Load sentence transformer for token-level embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers transformers")
        
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.get_sentence_embedding_dimension()  # Get base model name
        )
    
    def build_index(self):
        """Build token embeddings for documents."""
        if self.model is None:
            self.load_model()
        
        print(f"Encoding {len(self.documents)} documents...")
        
        self.doc_embeddings = []
        
        for i, doc in enumerate(self.documents):
            if (i + 1) % 5000 == 0:
                print(f"  Encoded {i+1}/{len(self.documents)}")
            
            doc_id = doc.get("doc_id", f"doc_{i}")
            title = doc.get("title", "")
            content = doc.get("content", "")
            text = f"{title} {content}".strip()
            
            # Tokenize
            tokens = self.tokenizer.encode(text, max_length=512, truncation=True)
            
            # Get token embeddings (using the model's pooling layer without pooling)
            # This is a workaround - properly requires ColBERT's token_embeddings
            embeddings = self.model.encode(text, convert_to_tensor=True)
            self.doc_embeddings.append((doc_id, embeddings))
        
        print(f"✓ Index built for {len(self.doc_embeddings)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MaxSim-like scoring."""
        if self.model is None or self.doc_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        scores = []
        for doc_id, doc_embedding in self.doc_embeddings:
            # Simple cosine similarity (not true MaxSim, but approximation)
            import torch
            score = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_embedding.unsqueeze(0)
            ).item()
            scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
