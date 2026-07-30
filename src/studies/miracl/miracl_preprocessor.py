import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import load_jsonl, save_jsonl, ensure_dir


class MIRACLPreprocessor:
    
    def __init__(self, language: str = "ro", 
                 input_dir: str = "data/miracl",
                 output_dir: str = "results/miracl"):
        self.language = language
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.corpus_dir = os.path.join(input_dir, "corpus")
        self.queries_dir = os.path.join(input_dir, "queries")
        self.qrels_dir = os.path.join(input_dir, "qrels")
        
        self.processed_dir = os.path.join(output_dir, "processed")
        self.splits_dir = os.path.join(output_dir, "splits")
        
        for d in [self.processed_dir, self.splits_dir]:
            ensure_dir(d)
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        corpus_file = os.path.join(
            self.corpus_dir, 
            f"miracl_{self.language}_corpus.jsonl"
        )
        
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        print(f"Loading corpus from {corpus_file}...")
        corpus = load_jsonl(corpus_file)
        print(f"Loaded {len(corpus)} documents")
        return corpus
    
    def load_queries(self) -> List[Dict[str, Any]]:
        queries_file = os.path.join(
            self.queries_dir,
            f"miracl_{self.language}_queries.jsonl"
        )
        
        if not os.path.exists(queries_file):
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        print(f"Loading queries from {queries_file}...")
        queries = load_jsonl(queries_file)
        print(f"Loaded {len(queries)} queries")
        return queries
    
    def load_qrels(self) -> Dict[str, List[str]]:
        qrels_file = os.path.join(
            self.qrels_dir,
            f"miracl_{self.language}_qrels.jsonl"
        )
        
        if not os.path.exists(qrels_file):
            raise FileNotFoundError(f"Qrels file not found: {qrels_file}")
        
        print(f"Loading qrels from {qrels_file}...")
        qrels_data = load_jsonl(qrels_file)
        
        qrels = {}
        for item in qrels_data:
            query_id = item.get("query_id")
            positive_passages = item.get("positive_passages", [])
            doc_ids = [p.get("docid") if isinstance(p, dict) else p 
                      for p in positive_passages]
            qrels[query_id] = doc_ids
        
        print(f"Loaded qrels for {len(qrels)} queries")
        return qrels
    
    def preprocess_corpus(self, corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"Preprocessing corpus...")
        
        processed = []
        for doc in tqdm(corpus, desc="Processing documents"):
            processed_doc = {
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "source": "miracl",
                "language": self.language,
                "original_id": doc.get("original_id", "")
            }
            processed.append(processed_doc)
        
        print(f"Processed {len(processed)} documents")
        return processed
    
    def preprocess_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"Preprocessing queries...")
        
        processed = []
        for query in tqdm(queries, desc="Processing queries"):
            processed_query = {
                "query_id": query.get("query_id", ""),
                "query": query.get("query", ""),
                "language": self.language,
                "original_id": query.get("original_id", "")
            }
            processed.append(processed_query)
        
        print(f"Processed {len(processed)} queries")
        return processed
    
    def split_data(self, corpus: List[Dict[str, Any]], 
                   queries: List[Dict[str, Any]],
                   qrels: Dict[str, List[str]],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[Dict, Dict, Dict]:
        
        print(f"Splitting data (train={train_ratio}, val={val_ratio})...")
        
        sorted_query_ids = sorted(qrels.keys())
        num_queries = len(sorted_query_ids)
        
        train_end = int(num_queries * train_ratio)
        val_end = int(num_queries * (train_ratio + val_ratio))
        
        train_ids = set(sorted_query_ids[:train_end])
        val_ids = set(sorted_query_ids[train_end:val_end])
        test_ids = set(sorted_query_ids[val_end:])
        
        splits = {
            "train": {"query_ids": list(train_ids), "size": len(train_ids)},
            "val": {"query_ids": list(val_ids), "size": len(val_ids)},
            "test": {"query_ids": list(test_ids), "size": len(test_ids)}
        }
        
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        return splits
    
    def run(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"MIRACL Preprocessor")
        print(f"{'='*60}")
        
        try:
            corpus = self.load_corpus()
            queries = self.load_queries()
            qrels = self.load_qrels()
            
            processed_corpus = self.preprocess_corpus(corpus)
            processed_queries = self.preprocess_queries(queries)
            
            corpus_out = os.path.join(self.processed_dir, f"corpus_processed.jsonl")
            queries_out = os.path.join(self.processed_dir, f"queries_processed.jsonl")
            qrels_out = os.path.join(self.processed_dir, f"qrels.json")
            
            save_jsonl(processed_corpus, corpus_out)
            save_jsonl(processed_queries, queries_out)
            
            with open(qrels_out, 'w', encoding='utf-8') as f:
                json.dump(qrels, f, indent=2, ensure_ascii=False)
            
            print(f"\nSaved processed files:")
            print(f"   - {corpus_out}")
            print(f"   - {queries_out}")
            print(f"   - {qrels_out}")
            
            splits = self.split_data(processed_corpus, processed_queries, qrels)
            
            splits_file = os.path.join(self.splits_dir, f"splits.json")
            with open(splits_file, 'w', encoding='utf-8') as f:
                json.dump(splits, f, indent=2)
            
            print(f"\nSaved splits to {splits_file}")
            
            summary = {
                "language": self.language,
                "corpus_size": len(processed_corpus),
                "queries_count": len(processed_queries),
                "qrels_count": len(qrels),
                "processed_dir": self.processed_dir,
                "splits_dir": self.splits_dir,
                "splits": splits
            }
            
            print(f"\n{'='*60}")
            print(f"Preprocessing Complete!")
            print(f"{'='*60}")
            print(f"Summary:")
            print(f"   Corpus: {summary['corpus_size']:,} documents")
            print(f"   Queries: {summary['queries_count']}")
            print(f"   Qrels: {summary['qrels_count']}")
            
            return summary
            
        except Exception as e:
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MIRACL dataset")
    parser.add_argument("--language", type=str, default="ro", help="Language code")
    parser.add_argument("--input-dir", type=str, default="data/miracl", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="results/miracl", help="Output directory")
    
    args = parser.parse_args()
    
    preprocessor = MIRACLPreprocessor(
        language=args.language,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    preprocessor.run()
