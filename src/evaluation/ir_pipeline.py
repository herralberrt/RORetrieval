#!/usr/bin/env python

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from utils import ensure_dir
from faiss_indexer import FaissIndexer
from triplet_generator import QueryGenerator, TripletGenerator
from quality_metrics import QualityMetrics
from data_splitter import DataSplitter
from evaluate_models import RetrievalEvaluator
from train_model import ModelTrainer


class IRPipeline:
    
    def __init__(self, corpus_path: str = "data/corpus/all_documents_combined.jsonl",
                 results_dir: str = "results"):
        self.corpus_path = corpus_path
        self.results_dir = results_dir
        self.results = {}
        self.start_time = time.time()
    
    def log(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] {message}")
    
    def step1_generate_triplets(self, num_queries: int = 1000, 
                               output_dir: str = None) -> Dict[str, Any]:
        
        self.log("STEP 1: Generating Triplets")
        
        output_dir = output_dir or os.path.join(self.results_dir, "triplets")
        
        self.log("  Building Faiss index...")
        indexer = FaissIndexer()
        indexer.load_model()
        indexer.build_index(self.corpus_path)
        
        self.log(f"  Generating {num_queries} synthetic queries...")
        query_gen = QueryGenerator()
        queries = query_gen.generate_synthetic_queries(count=num_queries)
        
        self.log("  Mining hard negatives and creating triplets...")
        triplet_gen = TripletGenerator(self.corpus_path)
        triplets = triplet_gen.generate_triplets(queries, results_dir=output_dir)
        
        self.results["triplets"] = {
            "count": len(triplets),
            "path": os.path.join(output_dir, "triplets.jsonl"),
            "queries": num_queries
        }
        
        self.log(f"  ✓ Generated {len(triplets)} triplets")
        return self.results["triplets"]
    
    def step2_quality_metrics(self, triplets_path: str = None,
                             output_dir: str = None) -> Dict[str, Any]:
        
        self.log("STEP 2: Evaluating Quality Metrics")
        
        triplets_path = triplets_path or self.results.get("triplets", {}).get("path")
        output_dir = output_dir or os.path.join(self.results_dir, "quality_metrics")
        
        if not triplets_path or not os.path.exists(triplets_path):
            self.log("  ✗ Triplets file not found")
            return {}
        
        self.log("  Computing quality metrics...")
        metrics = QualityMetrics()
        report = metrics.evaluate_dataset(triplets_path, output_dir)
        
        self.results["quality"] = {
            "valid_count": report.get("valid_triplets", 0),
            "total_count": report.get("total_triplets", 0),
            "validity_rate": report.get("validity_rate", 0),
            "report_path": os.path.join(output_dir, "quality_report.json")
        }
        
        self.log(f"  ✓ Quality metrics: {report['valid_triplets']}/{report['total_triplets']} valid")
        return self.results["quality"]
    
    def step3_data_split(self, triplets_path: str = None,
                        output_dir: str = None,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Dict[str, Any]:
        
        self.log("STEP 3: Splitting Data")
        
        triplets_path = triplets_path or self.results.get("triplets", {}).get("path")
        output_dir = output_dir or os.path.join(self.results_dir, "splits")
        
        if not triplets_path or not os.path.exists(triplets_path):
            self.log("  ✗ Triplets file not found")
            return {}
        
        self.log("  Splitting into train/val/test...")
        splitter = DataSplitter()
        paths = splitter.split_triplets(
            triplets_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            output_dir=output_dir
        )
        
        self.results["splits"] = {
            "train_path": paths["train"],
            "val_path": paths["val"],
            "test_path": paths["test"]
        }
        
        self.log("  ✓ Data split completed")
        return self.results["splits"]
    
    def step4_baseline_evaluation(self, test_triplets_path: str = None,
                                 output_dir: str = None) -> Dict[str, Any]:
        
        self.log("STEP 4: Baseline Evaluation")
        
        test_triplets_path = test_triplets_path or self.results.get("splits", {}).get("test_path")
        output_dir = output_dir or os.path.join(self.results_dir, "evaluation")
        
        if not test_triplets_path or not os.path.exists(test_triplets_path):
            self.log("  ✗ Test triplets file not found")
            return {}
        
        self.log("  Evaluating baseline model...")
        evaluator = RetrievalEvaluator(self.corpus_path)
        report = evaluator.evaluate_triplets(test_triplets_path, output_dir)
        
        self.results["baseline"] = {
            "ndcg": report["ndcg"]["mean"],
            "mrr": report["mrr"]["mean"],
            "precision": report["precision@10"]["mean"],
            "recall": report["recall@10"]["mean"],
            "map": report["map"]["mean"]
        }
        
        self.log(f"  ✓ Baseline nDCG@10: {report['ndcg']['mean']:.4f}")
        return self.results["baseline"]
    
    def step5_train_model(self, train_triplets_path: str = None,
                         val_triplets_path: str = None,
                         epochs: int = 3,
                         batch_size: int = 32,
                         output_dir: str = None) -> Dict[str, Any]:
        
        self.log("STEP 5: Training Model")
        
        train_triplets_path = train_triplets_path or self.results.get("splits", {}).get("train_path")
        val_triplets_path = val_triplets_path or self.results.get("splits", {}).get("val_path")
        output_dir = output_dir or os.path.join(self.results_dir, "trained_model")
        
        if not train_triplets_path or not os.path.exists(train_triplets_path):
            self.log("  ✗ Training triplets file not found")
            return {}
        
        self.log(f"  Training for {epochs} epochs with batch size {batch_size}...")
        trainer = ModelTrainer(corpus_path=self.corpus_path)
        trainer.train(
            train_triplets_path=train_triplets_path,
            val_triplets_path=val_triplets_path,
            epochs=epochs,
            batch_size=batch_size,
            output_dir=output_dir
        )
        
        self.results["training"] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "history_path": os.path.join(output_dir, "training_history.json")
        }
        
        self.log("  ✓ Model training completed")
        return self.results["training"]
    
    def generate_summary(self, output_path: str = None) -> str:
        
        output_path = output_path or os.path.join(self.results_dir, "pipeline_summary.json")
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": time.time() - self.start_time,
            "results": self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"\n{'='*60}")
        self.log("PIPELINE SUMMARY")
        self.log(f"{'='*60}")
        
        if "triplets" in self.results:
            self.log(f"✓ Step 1: Generated {self.results['triplets']['count']} triplets")
        
        if "quality" in self.results:
            rate = self.results["quality"]["validity_rate"]
            self.log(f"✓ Step 2: Quality metrics ({rate:.2%} valid)")
        
        if "splits" in self.results:
            self.log("✓ Step 3: Data split (70/15/15)")
        
        if "baseline" in self.results:
            ndcg = self.results["baseline"]["ndcg"]
            self.log(f"✓ Step 4: Baseline evaluation (nDCG: {ndcg:.4f})")
        
        if "training" in self.results:
            self.log(f"✓ Step 5: Model training ({self.results['training']['epochs']} epochs)")
        
        total_time = summary["total_time_seconds"]
        self.log(f"\nTotal Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        self.log(f"Summary saved to {output_path}")
        
        return output_path
    
    def run_full_pipeline(self, num_queries: int = 1000, epochs: int = 3):
        
        self.log("Starting IR Pipeline")
        self.log(f"Corpus: {self.corpus_path}")
        
        self.step1_generate_triplets(num_queries=num_queries)
        self.step2_quality_metrics()
        self.step3_data_split()
        self.step4_baseline_evaluation()
        self.step5_train_model(epochs=epochs)
        
        self.generate_summary()
        
        self.log("Pipeline completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete IR pipeline")
    parser.add_argument("--corpus", type=str, default="data/corpus/all_documents_combined.jsonl",
                       help="Corpus path")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--queries", type=int, default=1000,
                       help="Number of queries to generate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--step", type=int, default=0,
                       help="Run only specific step (0=all)")
    
    args = parser.parse_args()
    
    pipeline = IRPipeline(corpus_path=args.corpus, results_dir=args.results_dir)
    
    if args.step == 0:
        pipeline.run_full_pipeline(num_queries=args.queries, epochs=args.epochs)
    elif args.step == 1:
        pipeline.step1_generate_triplets(num_queries=args.queries)
    elif args.step == 2:
        pipeline.step2_quality_metrics()
    elif args.step == 3:
        pipeline.step3_data_split()
    elif args.step == 4:
        pipeline.step4_baseline_evaluation()
    elif args.step == 5:
        pipeline.step5_train_model(epochs=args.epochs)
