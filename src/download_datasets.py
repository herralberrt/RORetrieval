"""
Download and process datasets for Meeting 1.

Downloads from:
1. HuggingFace: recipes-ro
2. GitHub: RomanianNewsArticlesDataset
3. Common Crawl: CC-NEWS
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import save_jsonl, ensure_dir


class DatasetDownloader:
    
    def __init__(self, output_dir: str = "data/corpus"):
        """Initialize downloader."""
        self.output_dir = output_dir
        ensure_dir(output_dir)
        self.doc_counter = 0
    
    def download_recipes_huggingface(self) -> List[Dict[str, Any]]:

        print("\n Downloading recipes from HuggingFace...")
        
        try:
            from datasets import load_dataset
            
            # Load dataset
            dataset = load_dataset("BlackKakapo/recipes-ro", split="train")
            
            documents = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing recipes")):
                doc = {
                    "doc_id": f"recipe_{self.doc_counter:06d}",
                    "title": item.get("0", f"Recipe {idx}"),  # Column '0' = title
                    "ingredients": item.get("1", ""),         # Column '1' = ingredients
                    "content": item.get("2", ""),              # Column '2' = instructions/content
                    "source": "recipes-ro",
                    "original_id": idx
                }
                documents.append(doc)
                self.doc_counter += 1
            
            print(f" Downloaded {len(documents)} recipes")
            return documents
            
        except Exception as e:
            print(f" Error downloading recipes: {e}")
            return []
    
    def download_news_github(self) -> List[Dict[str, Any]]:

        print("\n Downloading news from GitHub...")
        
        try:
            # GitHub raw URL for RomanianNewsArticlesDataset
            urls = [
                "https://raw.githubusercontent.com/mhakan20/RomanianNewsArticlesDataset/master/articles.json"
            ]
            
            documents = []
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle different formats
                        items = data if isinstance(data, list) else data.get("articles", [])
                        
                        for idx, item in enumerate(tqdm(items, desc="Processing news")):
                            doc = {
                                "doc_id": f"news_{self.doc_counter:06d}",
                                "title": item.get("title", f"Article {idx}"),
                                "content": item.get("content", item.get("text", "")),
                                "url": item.get("url", ""),
                                "source": "news",
                                "original_id": idx
                            }
                            documents.append(doc)
                            self.doc_counter += 1
                        
                        print(f" Downloaded {len(documents)} news articles")
                        return documents
                        
                except Exception as e:
                    print(f"  Error with URL {url}: {e}")
                    continue
            
            return []
            
        except Exception as e:
            print(f" Error downloading news: {e}")
            return []
    
    def download_commoncrawl_index(self) -> List[Dict[str, Any]]:

        print("\n Downloading Common Crawl index...")
        
        try:
            # CC-NEWS index URL
            cc_url = "https://data.commoncrawl.org/crawl-data/CC-NEWS/2025/index.html"
            
            response = requests.get(cc_url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML to extract WARC file list
                import re
                
                # Find WARC files in HTML
                warc_pattern = r'href=["\']([^"\']*\.warc\.gz)["\']'
                warc_files = re.findall(warc_pattern, response.text)
                
                documents = []
                
                # Create metadata documents for available WARC files
                for warc_file in warc_files[:10]:  # Limit to first 10
                    doc = {
                        "doc_id": f"cc_{self.doc_counter:06d}",
                        "title": f"Common Crawl: {warc_file.split('/')[-1]}",
                        "content": f"WARC file: {warc_file}",
                        "warc_url": f"https://data.commoncrawl.org/{warc_file}",
                        "source": "commoncrawl",
                        "original_id": warc_file
                    }
                    documents.append(doc)
                    self.doc_counter += 1
                
                print(f" Found {len(documents)} Common Crawl WARC files")
                return documents
            
            return []
            
        except Exception as e:
            print(f" Error downloading CC-NEWS: {e}")
            return []
    
    def save_documents(self, documents: List[Dict[str, Any]], filename: str) -> None:

        output_path = os.path.join(self.output_dir, filename)
        save_jsonl(documents, output_path)
        print(f"   Saved to: {output_path}")
    
    def download_all(self) -> Dict[str, List[Dict[str, Any]]]:

        all_data = {}
        
        # Download recipes
        recipes = self.download_recipes_huggingface()
        all_data['recipes'] = recipes
        self.save_documents(recipes, "recipes_ro.jsonl")
        
        # Download news
        news = self.download_news_github()
        all_data['news'] = news
        self.save_documents(news, "news_ro.jsonl")
        
        # Download CC-NEWS metadata
        cc_news = self.download_commoncrawl_index()
        all_data['commoncrawl'] = cc_news
        self.save_documents(cc_news, "commoncrawl.jsonl")
        
        return all_data
    
    def combine_all_documents(self) -> None:
        """Combine all documents into single file with numbered docs."""
        print("\n Combining all documents...")
        
        try:
            all_docs = []
            
            # Load from individual files
            for filename in ["recipes_ro.jsonl", "news_ro.jsonl", "commoncrawl.jsonl"]:
                filepath = os.path.join(self.output_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                all_docs.append(json.loads(line))
            
            print(f"   Total documents: {len(all_docs)}")
            
            # Save combined file
            combined_path = os.path.join(self.output_dir, "all_documents_numbered.jsonl")
            self.save_documents(all_docs, "all_documents_numbered.jsonl")
            
            # Print summary
            print(f"\n Summary:")
            print(f"   Recipes: {len([d for d in all_docs if d['source'] == 'recipes-ro'])}")
            print(f"   News: {len([d for d in all_docs if d['source'] == 'news'])}")
            print(f"   Common Crawl: {len([d for d in all_docs if d['source'] == 'commoncrawl'])}")
            print(f"   TOTAL: {len(all_docs)}")
            
        except Exception as e:
            print(f" Error combining documents: {e}")


def main():

    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process datasets for Meeting 1")

    parser.add_argument('--output', type=str, default='data/corpus', help='Output directory')

    parser.add_argument('--combine', action='store_true', help='Combine all documents into single file')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Dataset Downloader - Meeting 1")
    print("="*60)
    
    downloader = DatasetDownloader(output_dir=args.output)
    
    all_data = downloader.download_all()
    
    if args.combine:
        downloader.combine_all_documents()
    
    print("\n" + "="*60)
    print("   Download complete!")
    print("="*60)
    print(f"\n  Files saved to: {args.output}/")
    print(f"  - recipes_ro.jsonl")
    print(f"  - news_ro.jsonl")
    print(f"  - commoncrawl.jsonl")
    print(f"  - all_documents_numbered.jsonl (if --combine used)")
    print()


if __name__ == '__main__':
    main()
