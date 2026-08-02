import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys
import argparse

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))
from utils import ensure_dir, save_json

try:
    # Package-relative import (works as `src.studies.miracl`).
    from .miracl_downloader import MIRACLDownloader
    from .miracl_preprocessor import MIRACLPreprocessor
except ImportError:
    # Flat import: this directory is on sys.path (how run.sh invokes it).
    from miracl_downloader import MIRACLDownloader
    from miracl_preprocessor import MIRACLPreprocessor


class MIRACLPipeline:
    
    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        self.config = config
        self.verbose = verbose
        
        self.language = config.get("language", "ro")
        self.data_dir = config.get("data_dir", "data/miracl")
        self.results_dir = config.get("results_dir", "results/miracl")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        
        ensure_dir(self.data_dir)
        ensure_dir(self.results_dir)
        ensure_dir(self.logs_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"pipeline_{timestamp}.log")
        
        self.log(f"Pipeline initialized: {timestamp}")
        self.log(f"Config: {json.dumps(config, indent=2)}")
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def step_download(self) -> Dict[str, Any]:
        self.log("="*60)
        self.log("STEP 0: Download MIRACL Dataset", "INFO")
        self.log("="*60)
        
        try:
            downloader = MIRACLDownloader(
                language=self.language,
                output_dir=self.data_dir
            )
            summary = downloader.download_all()
            
            self.log(f"Download complete: {json.dumps(summary)}", "SUCCESS")
            return summary
            
        except Exception as e:
            self.log(f"Download failed: {str(e)}", "ERROR")
            raise
    
    def step_preprocess(self) -> Dict[str, Any]:
        self.log("="*60)
        self.log("STEP 0.5: Preprocess MIRACL Data", "INFO")
        self.log("="*60)
        
        try:
            preprocessor = MIRACLPreprocessor(
                language=self.language,
                input_dir=self.data_dir,
                output_dir=self.results_dir
            )
            summary = preprocessor.run()
            
            self.log(f"Preprocessing complete: {json.dumps(summary)}", "SUCCESS")
            return summary
            
        except Exception as e:
            self.log(f"Preprocessing failed: {str(e)}", "ERROR")
            raise
    
    def step_task1(self, corpus_file: str, queries_file: str) -> Dict[str, Any]:
        self.log("="*60)
        self.log("STEP 1: Generate Queries (Task 1)", "INFO")
        self.log("="*60)
        
        try:
            self.log(f"Input corpus: {corpus_file}")
            self.log(f"Output queries file: {queries_file}")

            from task1_prompt import generate_llm_queries
            from llm_client import get_client

            client = get_client()
            self.log(client.describe())

            if not client.available:
                self.log("Skipping Task 1: no API key. Put ANTHROPIC_API_KEY or "
                         "OPENAI_API_KEY in .env to enable it.", "WARNING")
                return {
                    "status": "skipped",
                    "reason": client.unavailable_reason,
                }

            task_cfg = self.config.get("task_config", {}).get("task1_queries", {})
            generate_llm_queries(
                input_file=corpus_file,
                output_file=queries_file,
                versions=[task_cfg.get("prompt_version", "v1").lower()],
                temperatures=[task_cfg.get("temperature", 0.8)],
                limit=task_cfg.get("limit"),
                model=task_cfg.get("model"),
            )

            return {
                "status": "success",
                "output": queries_file,
                "model": client.model,
            }

        except Exception as e:
            self.log(f"Task 1 failed: {str(e)}", "ERROR")
            raise
    
    def step_task2(self, queries_file: str, corpus_file: str) -> Dict[str, Any]:
        self.log("="*60)
        self.log("STEP 2: Create Triplets (Task 2)", "INFO")
        self.log("="*60)
        
        try:
            self.log(f"Input queries: {queries_file}")
            self.log(f"Input corpus: {corpus_file}")
            
            self.log("Task 2 would create triplets: (query, positive_doc, negative_docs)")
            self.log("   (Requires similarity computation and hard negative mining)")
            
            return {
                "status": "configured",
                "message": "Task 2 ready for execution (requires embeddings)"
            }
            
        except Exception as e:
            self.log(f"Task 2 failed: {str(e)}", "ERROR")
            raise
    
    def step_task3(self, triplets_file: str) -> Dict[str, Any]:
        self.log("="*60)
        self.log("STEP 3: Filter & Evaluate Triplets (Task 3)", "INFO")
        self.log("="*60)
        
        try:
            self.log(f"Input triplets: {triplets_file}")
            
            self.log("Task 3 would filter triplets based on quality metrics")
            self.log("   (Overlap, diversity, entailment scores)")
            
            return {
                "status": "configured",
                "message": "Task 3 ready for execution (requires quality metrics)"
            }
            
        except Exception as e:
            self.log(f"Task 3 failed: {str(e)}", "ERROR")
            raise
    
    def run(self, skip_download: bool = False, skip_preprocess: bool = False) -> Dict[str, Any]:
        
        self.log("Starting MIRACL Pipeline")
        
        execution_start = datetime.now()
        results = {"steps": {}, "timestamp": execution_start.isoformat()}
        
        try:
            if not skip_download:
                results["steps"]["download"] = self.step_download()
            else:
                self.log("Skipping download")
            
            if not skip_preprocess:
                preprocess_result = self.step_preprocess()
                results["steps"]["preprocess"] = preprocess_result
                
                corpus_file = os.path.join(
                    preprocess_result.get("processed_dir", ""),
                    "corpus_processed.jsonl"
                )
                queries_file = os.path.join(
                    preprocess_result.get("processed_dir", ""),
                    "queries_processed.jsonl"
                )
            else:
                self.log("Skipping preprocessing")
                corpus_file = os.path.join(self.results_dir, "processed/corpus_processed.jsonl")
                queries_file = os.path.join(self.results_dir, "processed/queries_processed.jsonl")
            
            if self.config.get("run_task1", True):
                task1_output = os.path.join(self.results_dir, "task1_queries.jsonl")
                results["steps"]["task1"] = self.step_task1(corpus_file, task1_output)
            
            if self.config.get("run_task2", True):
                task1_output = os.path.join(self.results_dir, "task1_queries.jsonl")
                results["steps"]["task2"] = self.step_task2(task1_output, corpus_file)
            
            if self.config.get("run_task3", True):
                task2_output = os.path.join(self.results_dir, "task2_triplets.jsonl")
                results["steps"]["task3"] = self.step_task3(task2_output)
            
            execution_end = datetime.now()
            duration = (execution_end - execution_start).total_seconds()
            
            results["status"] = "success"
            results["duration_seconds"] = duration
            results["log_file"] = self.log_file
            
            self.log("="*60)
            self.log("Pipeline Complete!", "SUCCESS")
            self.log("="*60)
            self.log(f"Duration: {duration:.2f} seconds")
            self.log(f"Log file: {self.log_file}")
            
            return results
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            
            self.log("="*60)
            self.log(f"Pipeline Failed: {str(e)}", "ERROR")
            self.log("="*60)
            
            return results


def load_config(config_file: str) -> Dict[str, Any]:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="MIRACL Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Configuration file")
    parser.add_argument("--language", type=str, default="ro",
                       help="Language code (default: ro)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download step")
    parser.add_argument("--skip-preprocess", action="store_true",
                       help="Skip preprocessing step")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {
            "language": args.language,
            "data_dir": "data/miracl",
            "results_dir": "results/miracl",
            "run_task1": True,
            "run_task2": True,
            "run_task3": True
        }
        print(f"Using default config (create {args.config} to customize)")
    
    pipeline = MIRACLPipeline(config, verbose=args.verbose)
    results = pipeline.run(
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess
    )
    
    results_file = os.path.join(config["results_dir"], "pipeline_results.json")
    save_json(results, results_file)
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
