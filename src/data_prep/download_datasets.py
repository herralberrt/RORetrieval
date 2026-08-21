import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import sys
from abc import ABC, abstractmethod

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import save_jsonl, ensure_dir


class CategoryDatasetManager:
    """Manages dynamic category directories and dataset organization."""
    
    def __init__(self, categories_dir: str = "data/categories"):
        self.categories_dir = categories_dir
        ensure_dir(categories_dir)
        self.category_counters = {}
    
    def get_category_dir(self, category: str) -> str:
        """Get or create category directory."""
        category_dir = os.path.join(self.categories_dir, category)
        ensure_dir(category_dir)
        return category_dir
    
    def save_category_documents(
        self, 
        documents: List[Dict[str, Any]], 
        category: str, 
        filename: str = None
    ) -> str:
        """Save documents to category directory."""
        if filename is None:
            filename = f"{category}.jsonl"
        
        category_dir = self.get_category_dir(category)
        filepath = os.path.join(category_dir, filename)
        save_jsonl(documents, filepath)
        print(f"Saved {len(documents)} documents to: {filepath}")
        return filepath
    
    def get_next_doc_id(self, category: str, prefix: str = None) -> str:
        """Generate next document ID for category."""
        if category not in self.category_counters:
            self.category_counters[category] = 0
        
        if prefix is None:
            prefix = category[:3]  # Use first 3 letters of category
        
        doc_id = f"{prefix}_{self.category_counters[category]:06d}"
        self.category_counters[category] += 1
        return doc_id


class BaseDatasetDownloader(ABC):
    """Base class for dataset downloaders."""
    
    def __init__(self, manager: CategoryDatasetManager):
        self.manager = manager
        self.category = self._get_category_name()
        self.limit = None  # Can be overridden per instance
    
    @abstractmethod
    def _get_category_name(self) -> str:
        """Return the category name for this dataset."""
        pass
    
    @abstractmethod
    def download(self) -> List[Dict[str, Any]]:
        """Download and process dataset."""
        pass
    
    def set_limit(self, limit: int) -> None:
        """Set document limit for this downloader."""
        self.limit = limit if limit > 0 else None
    
    def save(self, documents: List[Dict[str, Any]]) -> None:
        """Save documents to category."""
        self.manager.save_category_documents(documents, self.category)


class RecipesDownloader(BaseDatasetDownloader):
    """Downloader for Romanian recipes dataset."""
    
    def _get_category_name(self) -> str:
        return "recipes"
    
    def download(self) -> List[Dict[str, Any]]:
        print("\nDownloading recipes from HuggingFace...")
        print("Dataset: BlackKakapo/recipes-ro")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("BlackKakapo/recipes-ro", split="train")
            
            documents = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing recipes")):
                doc = {
                    "doc_id": self.manager.get_next_doc_id(self.category, "rec"),
                    "title": item.get("0", f"Recipe {idx}"),
                    "ingredients": item.get("1", ""),
                    "content": item.get("2", ""),
                    "source": "BlackKakapo/recipes-ro",
                    "original_id": idx
                }
                documents.append(doc)
            
            print(f"Downloaded {len(documents)} recipes")
            return documents
            
        except Exception as e:
            print(f"Error downloading recipes: {e}")
            raise


class RoStoriesDownloader(BaseDatasetDownloader):
    """Downloader for Romanian Stories dataset."""
    
    def _get_category_name(self) -> str:
        return "stories"
    
    def download(self) -> List[Dict[str, Any]]:
        print("\nDownloading Romanian stories from HuggingFace...")
        print("Dataset: readerbench/ro-stories")
        print("Note: Dataset contains story paragraphs from Romanian authors")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("readerbench/ro-stories", split="train")
            
            documents = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing stories")):
                # Available fields: author, title, paragraph, word_count
                author = item.get("author", "Unknown")
                title = item.get("title", f"Story {idx}")
                paragraph = item.get("paragraph", "")  # Main content is in "paragraph" field
                word_count = item.get("word_count", 0)
                
                doc = {
                    "doc_id": self.manager.get_next_doc_id(self.category, "sto"),
                    "title": title,
                    "author": author,
                    "content": paragraph,
                    "word_count": word_count,
                    "source": "readerbench/ro-stories",
                    "original_id": idx
                }
                documents.append(doc)
            
            print(f"Downloaded {len(documents)} story paragraphs")
            return documents
            
        except Exception as e:
            print(f"Error downloading stories: {e}")
            raise


class RoTextSummarizationDownloader(BaseDatasetDownloader):
    """Downloader for Romanian Text Summarization dataset."""
    
    def _get_category_name(self) -> str:
        return "summarization"
    
    def download(self) -> List[Dict[str, Any]]:
        print("\nDownloading Romanian text summarization from HuggingFace...")
        print("Dataset: readerbench/ro-text-summarization")
        print("Note: Dataset contains articles with summaries (~65k total)")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("readerbench/ro-text-summarization", split="train")
            
            # Apply limit if set (default ALL documents if not specified)
            limit = self.limit if self.limit else -1
            if limit > 0 and limit < len(dataset):
                print(f"Limiting to first {limit} documents (total available: {len(dataset)})")
                dataset = dataset.select(range(min(limit, len(dataset))))
            else:
                print(f"Downloading all {len(dataset)} documents")
            
            documents = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing summarization")):
                # Available fields: Category, Title, Content, Summary, href, Source
                category = item.get("Category", "uncategorized")
                title = item.get("Title", f"Article {idx}")
                content = item.get("Content", "")
                summary = item.get("Summary", "")
                source = item.get("Source", "unknown")
                href = item.get("href", "")
                
                doc = {
                    "doc_id": self.manager.get_next_doc_id(self.category, "sum"),
                    "title": title,
                    "content": content,
                    "summary": summary,
                    "category": category,
                    "source_url": href,
                    "source": f"readerbench/ro-text-summarization ({source})",
                    "original_id": idx
                }
                documents.append(doc)
            
            print(f"Downloaded {len(documents)} summarization articles")
            return documents
            
        except Exception as e:
            print(f"Error downloading summarization: {e}")
            raise


class RomanianNewsDownloader(BaseDatasetDownloader):
    """Downloader for Romanian News Articles from multiple outlets.
    
    Downloads ~150k articles from 10 Romanian news outlets:
    - Adevărul, Digi24, Libertatea, Mediafax, ProTV, Ziarul Financiar,
    - EVZ, Cotidianul, Aleph, Realitatea
    
    Each outlet is saved as a separate category with title, text, and summary.
    """
    
    def _get_category_name(self) -> str:
        return "news"
    
    def download(self) -> List[Dict[str, Any]]:
        print("\nDownloading Romanian News Articles...")
        print("Repository: mhakan20/RomanianNewsArticlesDataset")
        print("Note: Contains articles from 10 Romanian news outlets")
        
        import tempfile
        import shutil
        import subprocess
        
        documents = []
        temp_dir = None
        
        try:
            # Clone repository to temp directory
            temp_dir = tempfile.mkdtemp(prefix="ro_news_")
            print(f"Cloning repository to {temp_dir}...")
            
            result = subprocess.run(
                ["git", "clone", "--depth=1", 
                 "https://github.com/mhakan20/RomanianNewsArticlesDataset.git",
                 temp_dir],
                capture_output=True,
                # The repo carries ~350MB of JSON; 2 minutes is not enough on a
                # shared cluster link.
                timeout=int(os.environ.get("NEWS_CLONE_TIMEOUT", 900))
            )
            
            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr.decode()}")
            
            datasets_dir = os.path.join(temp_dir, "datasets")
            
            # Map JSON files to news outlets
            news_sources = {
                'adevarul': 'news_adevarul.json',
                'digi24': 'news_digi.json',
                'libertatea': 'news_libertatea.json',
                'mediafax': 'news_mediafax.json',
                'protv': 'news_protv.json',
                'zf': 'news_zf.json',
                'evz': 'news_evz.json',
                'cotidianul': 'news_cotidianul.json',
                'aleph': 'news_aleph.json',
                'realitatea': 'news_realitatea.json',
            }
            
            total_docs = 0
            
            # Process each news source
            for source_key, json_file in news_sources.items():
                json_path = os.path.join(datasets_dir, json_file)
                
                if not os.path.exists(json_path):
                    print(f"⚠ Warning: {json_file} not found, skipping")
                    continue
                
                print(f"\nProcessing {source_key.upper()}...")
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    # Apply limit if set
                    if self.limit and self.limit > 0:
                        articles = articles[:self.limit]
                    
                    # Save to source-specific category
                    source_docs = []
                    for idx, article in enumerate(tqdm(articles, desc=f"Processing {source_key}")):
                        doc = {
                            "doc_id": self.manager.get_next_doc_id(source_key, source_key[:3]),
                            "title": article.get("title", f"Article {idx}"),
                            "text": article.get("text", ""),
                            "summary": article.get("summary", ""),
                            "source": source_key,
                            "publication": source_key,
                            "original_id": idx
                        }
                        source_docs.append(doc)
                    
                    # Save to separate category per source
                    self.manager.save_category_documents(
                        source_docs,
                        source_key,
                        filename=f"news_{source_key}.jsonl"
                    )
                    
                    total_docs += len(source_docs)
                    print(f"  → {len(source_docs)} articles from {source_key}")
                    
                    documents.extend(source_docs)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing {json_file}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing {source_key}: {e}")
                    continue
            
            print(f"\nTotal news articles downloaded: {total_docs}")
            return documents
            
        except Exception as e:
            print(f"Error downloading news articles: {e}")
            raise
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory")
                except Exception as e:
                    print(f"Warning: Could not cleanup temp directory: {e}")


class DatasetDownloader:
    """Main orchestrator for dataset downloads."""
    
    def __init__(self):
        self.manager = CategoryDatasetManager()
        self.downloaders = {
            'recipes': RecipesDownloader(self.manager),
            'stories': RoStoriesDownloader(self.manager),
            'summarization': RoTextSummarizationDownloader(self.manager),
            'news': RomanianNewsDownloader(self.manager),
        }
    
    def download_dataset(self, name: str, limit: int = None) -> Optional[List[Dict[str, Any]]]:
        """Download a specific dataset.
        
        Args:
            name: Dataset name
            limit: Maximum number of documents to download (None = all, -1 = all, >0 = limit)
        """
        if name not in self.downloaders:
            print(f"Error: Dataset '{name}' not found.")
            print(f"Available datasets: {', '.join(self.downloaders.keys())}")
            return None
        
        downloader = self.downloaders[name]
        
        # Set limit if provided
        if limit is not None:
            downloader.set_limit(limit)
        
        documents = downloader.download()
        downloader.save(documents)
        return documents
    
    def download_all(self, limit: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """Download all available datasets."""
        all_data = {}
        
        for name in self.downloaders.keys():
            try:
                documents = self.download_dataset(name, limit=limit)
                if documents:
                    all_data[name] = documents
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
        
        return all_data
    
    def list_available_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self.downloaders.keys())
    
    def combine_category_documents(self, category: str = None) -> None:
        """Combine all documents from a specific category."""
        if category:
            print(f"\nCombining documents from category: {category}")
            category_dir = self.manager.get_category_dir(category)
            
            all_docs = []
            for filename in os.listdir(category_dir):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(category_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    all_docs.append(json.loads(line))
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
            
            if all_docs:
                combined_path = os.path.join(category_dir, f"{category}_combined.jsonl")
                save_jsonl(all_docs, combined_path)
                print(f"Combined {len(all_docs)} documents to: {combined_path}")
        else:
            print("\nCombining all documents from all categories...")
            all_docs = []
            
            for category_name in os.listdir(self.manager.categories_dir):
                category_path = os.path.join(self.manager.categories_dir, category_name)
                if os.path.isdir(category_path):
                    for filename in os.listdir(category_path):
                        if filename.endswith('.jsonl') and not filename.endswith('_combined.jsonl'):
                            filepath = os.path.join(category_path, filename)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        if line.strip():
                                            all_docs.append(json.loads(line))
                            except Exception as e:
                                print(f"Error reading {filepath}: {e}")
            
            if all_docs:
                combined_path = os.path.join(self.manager.categories_dir, "all_documents.jsonl")
                save_jsonl(all_docs, combined_path)
                print(f"\nSummary:")
                sources = {}
                for doc in all_docs:
                    source = doc.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                for source, count in sorted(sources.items()):
                    print(f"  {source}: {count} documents")
                print(f"  TOTAL: {len(all_docs)} documents")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process datasets")
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specific dataset to download (e.g., recipes, stories, summarization)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum documents to download (-1 for all, default varies by dataset)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--combine',
        type=str,
        nargs='?',
        const='all',
        help='Combine documents from category or all categories'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Dataset Downloader - Dynamic Categories")
    print("="*60)
    
    downloader = DatasetDownloader()
    
    if args.list:
        print("\nAvailable datasets:")
        for ds in downloader.list_available_datasets():
            print(f"  - {ds}")
    elif args.dataset:
        try:
            downloader.download_dataset(args.dataset, limit=args.limit)
        except Exception as e:
            print(f"Error: {e}")
    elif args.combine:
        category = None if args.combine == 'all' else args.combine
        downloader.combine_category_documents(category)
    else:
        # Default: download all datasets
        all_data = downloader.download_all(limit=args.limit)
        print("\n" + "="*60)
        print("Download complete!")
        print("="*60)
        for name, docs in all_data.items():
            print(f"{name}: {len(docs)} documents")


if __name__ == '__main__':
    main()

