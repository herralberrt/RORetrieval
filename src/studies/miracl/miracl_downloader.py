import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import save_jsonl, ensure_dir


class MIRACLDownloader:
    
    def __init__(self, language: str = "ro", output_dir: str = "data/miracl"):
        self.language = language
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        self.corpus_dir = os.path.join(output_dir, "corpus")
        self.queries_dir = os.path.join(output_dir, "queries")
        self.qrels_dir = os.path.join(output_dir, "qrels")
        
        for d in [self.corpus_dir, self.queries_dir, self.qrels_dir]:
            ensure_dir(d)
    
    def download_corpus(self) -> List[Dict[str, Any]]:
        print(f"Downloading MIRACL corpus for language: {self.language}")
        
        try:
            from datasets import load_dataset
            
            dataset_name = f"miracl/miracl-corpus-{self.language}"
            corpus = load_dataset(dataset_name, split="train")
            
            documents = []
            for idx, item in enumerate(tqdm(corpus, desc="Processing corpus")):
                doc = {
                    "doc_id": item.get("docid", f"doc_{idx:06d}"),
                    "title": item.get("title", ""),
                    "content": item.get("text", ""),
                    "source": "miracl",
                    "language": self.language,
                    "original_id": idx
                }
                documents.append(doc)
            
            output_path = os.path.join(self.corpus_dir, f"miracl_{self.language}_corpus.jsonl")
            save_jsonl(documents, output_path)
            print(f"Saved {len(documents)} documents to {output_path}")
            
            return documents
            
        except Exception as e:
            print(f"Error downloading corpus: {e}")
            return []
    
    def download_queries(self) -> List[Dict[str, Any]]:
        print(f"Downloading MIRACL queries for language: {self.language}")
        
        try:
            from datasets import load_dataset
            
            dataset_name = f"miracl/miracl-{self.language}"
            queries_dataset = load_dataset(dataset_name, split="dev")
            
            queries = []
            for idx, item in enumerate(tqdm(queries_dataset, desc="Processing queries")):
                query = {
                    "query_id": item.get("query_id", f"q_{idx:06d}"),
                    "query": item.get("query", ""),
                    "language": self.language,
                    "original_id": idx
                }
                queries.append(query)
            
            output_path = os.path.join(self.queries_dir, f"miracl_{self.language}_queries.jsonl")
            save_jsonl(queries, output_path)
            print(f"Saved {len(queries)} queries to {output_path}")
            
            return queries
            
        except Exception as e:
            print(f"Error downloading queries: {e}")
            return []
    
    def download_qrels(self) -> Dict[str, List[Dict[str, Any]]]:
        print(f"Downloading MIRACL qrels for language: {self.language}")
        
        try:
            from datasets import load_dataset
            
            dataset_name = f"miracl/miracl-{self.language}"
            qrels_dataset = load_dataset(dataset_name, split="dev")
            
            qrels = {}
            qrels_flat = []
            
            for idx, item in enumerate(tqdm(qrels_dataset, desc="Processing qrels")):
                query_id = item.get("query_id", f"q_{idx:06d}")
                positive_passages = item.get("positive_passages", [])
                
                qrel = {
                    "query_id": query_id,
                    "positive_passages": positive_passages,
                    "num_positives": len(positive_passages),
                    "language": self.language
                }
                qrels_flat.append(qrel)
                
                qrels[query_id] = positive_passages
            
            output_path = os.path.join(self.qrels_dir, f"miracl_{self.language}_qrels.jsonl")
            save_jsonl(qrels_flat, output_path)
            print(f"Saved qrels for {len(qrels)} queries to {output_path}")
            
            return qrels
            
        except Exception as e:
            print(f"Error downloading qrels: {e}")
            return {}
    
    def download_all(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"MIRACL Dataset Downloader - Language: {self.language.upper()}")
        print(f"{'='*60}")
        
        corpus = self.download_corpus()
        queries = self.download_queries()
        qrels = self.download_qrels()
        
        summary = {
            "language": self.language,
            "corpus_size": len(corpus),
            "queries_count": len(queries),
            "qrels_count": len(qrels),
            "output_dir": self.output_dir,
            "corpus_dir": self.corpus_dir,
            "queries_dir": self.queries_dir,
            "qrels_dir": self.qrels_dir
        }
        
        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"   Language: {summary['language']}")
        print(f"   Corpus size: {summary['corpus_size']:,} documents")
        print(f"   Queries: {summary['queries_count']}")
        print(f"   Qrels: {summary['qrels_count']}")
        print(f"   Output: {summary['output_dir']}")
        
        return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MIRACL dataset")
    parser.add_argument("--language", type=str, default="ro", help="Language code (default: ro)")
    parser.add_argument("--output-dir", type=str, default="data/miracl", help="Output directory")
    
    args = parser.parse_args()
    
    downloader = MIRACLDownloader(language=args.language, output_dir=args.output_dir)
    downloader.download_all()
