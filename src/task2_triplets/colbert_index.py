"""
Late-interaction (ColBERT-style) retrieval over the Romanian corpus.

A dense bi-encoder collapses a document into one vector, so "Rădoi a semnat cu
Craiova" and "Craiova a semnat cu Rădoi" land in nearly the same place and a
long document's specifics get averaged away. Late interaction keeps one vector
per *token* and scores with MaxSim - for every query token, the best-matching
document token, summed:

    score(q, d) = Σ_i max_j  Eq_i · Ed_j          (both L2-normalised)

That is the ColBERT formulation (arxiv 2004.12832). What this file is *not* is
a trained ColBERT checkpoint: `colbert-ir` is not in the image and there is no
ColBERT trained for Romanian, so the token embeddings come from a multilingual
sentence-transformer that was trained for mean-pooled similarity, not for late
interaction. Calling it "ColBERT" would overclaim - it is late-interaction
scoring over an off-the-shelf multilingual encoder, and it earns its place by
ranking differently from both BM25 and the mean-pooled dense run, not by being
a faithful ColBERT.

Two stages, because MaxSim over a whole corpus per query is not affordable:

1.  **Candidates** from the mean-pooled vectors - one matmul over an
    87k x 384 matrix, milliseconds for a batch of queries.
2.  **Rerank** those candidates with MaxSim over their token embeddings.

The whole token tensor is held on the GPU (87k docs x 128 tokens x 384 dims in
fp16 is ~8.6 GB, against 80 GB on an H200), so the rerank is a single einsum
with no host round-trip. On CPU the same code runs, slowly - use
`--doc-max-tokens 64` and a corpus subset if you have to.
"""

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LateInteractionIndex:
    """Token-level index with MaxSim scoring and a mean-pooled first stage."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 doc_max_tokens: int = 128,
                 query_max_tokens: int = 32,
                 batch_size: int = 256,
                 dtype: str = "float16"):
        import torch
        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_max_tokens = doc_max_tokens
        self.query_max_tokens = query_max_tokens
        self.batch_size = batch_size
        # fp16 halves the resident tensor and MaxSim is a ranking, not a
        # calibrated score, so the precision loss does not change the order.
        self.dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32

        self.model = None
        self.tokenizer = None
        self.doc_tokens = None      # (n_docs, doc_max_tokens, dim), normalised
        self.doc_mask = None        # (n_docs, doc_max_tokens) bool
        self.doc_pooled = None      # (n_docs, dim), normalised
        self.dim = None

    # ------------------------------------------------------------------ model
    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        print(f"  loading {self.model_name} on {self.device} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)
        self.dim = self.model.config.hidden_size

    def _encode(self, texts: Sequence[str], max_tokens: int):
        """`(token_embeddings, mask)` for one batch, L2-normalised."""
        torch = self.torch
        batch = self.tokenizer(list(texts), padding="max_length", truncation=True,
                               max_length=max_tokens, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch).last_hidden_state          # (b, L, d)
        mask = batch["attention_mask"].bool()
        # Normalising here means MaxSim is a plain dot product later.
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out, mask

    # ------------------------------------------------------------------ index
    def build(self, texts: Sequence[str]) -> None:
        """Encode the corpus once: token embeddings plus mean-pooled vectors."""
        torch = self.torch
        if self.model is None:
            self.load_model()
        n = len(texts)
        self.doc_tokens = torch.zeros((n, self.doc_max_tokens, self.dim),
                                      dtype=self.dtype, device=self.device)
        self.doc_mask = torch.zeros((n, self.doc_max_tokens),
                                    dtype=torch.bool, device=self.device)
        pooled = torch.zeros((n, self.dim), dtype=self.dtype, device=self.device)

        for start in range(0, n, self.batch_size):
            chunk = texts[start:start + self.batch_size]
            emb, mask = self._encode(chunk, self.doc_max_tokens)
            end = start + len(chunk)
            self.doc_tokens[start:end] = emb.to(self.dtype)
            self.doc_mask[start:end] = mask
            # Mean over real tokens only; the padded rows are already zero.
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled[start:end] = (emb.sum(dim=1) / counts).to(self.dtype)
            if (end // self.batch_size) % 40 == 0:
                print(f"    encoded {end}/{n}", flush=True)

        self.doc_pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1).to(self.dtype)
        gb = self.doc_tokens.element_size() * self.doc_tokens.nelement() / 1e9
        print(f"  index: {n} docs x {self.doc_max_tokens} tokens x {self.dim} dims "
              f"({gb:.1f} GB on {self.device})")

    # ----------------------------------------------------------------- search
    def encode_queries(self, queries: Sequence[str]):
        """`(token_embeddings, mask)` for a batch of queries."""
        return self._encode(queries, self.query_max_tokens)

    def search(self, queries: Sequence[str], top_k: int,
               first_stage: int = 1000,
               allowed: Optional["np.ndarray"] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Late-interaction search.

        `allowed` restricts the search to a subset of document rows - used to
        keep negatives inside the positive's own document type, the same way the
        BM25 builder does. Returns `(indices, scores)`, both (n_queries, top_k),
        best first.
        """
        torch = self.torch
        q_emb, q_mask = self.encode_queries(queries)          # (b, Lq, d)

        pooled = self.doc_pooled
        index_map = None
        if allowed is not None:
            index_map = torch.as_tensor(np.asarray(allowed), device=self.device,
                                        dtype=torch.long)
            pooled = pooled[index_map]

        # Stage 1: mean-pooled cosine, to cut the corpus down to a shortlist.
        q_pooled = torch.nn.functional.normalize(
            (q_emb.sum(dim=1) / q_mask.sum(dim=1, keepdim=True).clamp(min=1)).float(),
            p=2, dim=-1).to(self.dtype)
        coarse = q_pooled @ pooled.T                          # (b, n_allowed)
        k1 = min(first_stage, coarse.shape[1])
        cand = coarse.topk(k1, dim=1).indices                 # (b, k1)

        # Stage 2: MaxSim over the shortlist's token embeddings.
        results_i, results_s = [], []
        for row in range(len(queries)):
            local = cand[row]
            rows = index_map[local] if index_map is not None else local
            d_tok = self.doc_tokens[rows]                     # (k1, Ld, d)
            d_mask = self.doc_mask[rows]                      # (k1, Ld)
            q = q_emb[row][q_mask[row]].to(self.dtype)        # (nq, d)
            # (k1, nq, Ld): every query token against every document token.
            sim = torch.einsum("qd,kld->kql", q, d_tok)
            sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
            score = sim.max(dim=2).values.sum(dim=1)          # MaxSim
            k2 = min(top_k, score.shape[0])
            best = score.topk(k2)
            results_i.append(rows[best.indices].detach().cpu().numpy())
            results_s.append(best.values.float().detach().cpu().numpy())

        pad_i = np.full((len(queries), top_k), -1, dtype=np.int64)
        pad_s = np.zeros((len(queries), top_k), dtype=np.float32)
        for r, (idx, sc) in enumerate(zip(results_i, results_s)):
            pad_i[r, :len(idx)] = idx
            pad_s[r, :len(sc)] = sc
        return pad_i, pad_s
