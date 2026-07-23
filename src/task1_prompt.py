"""
TASK 1: Generate LLM Prompts for Query Generation

This script creates two versions of prompts:
- V1: Simple (title + article) → queries
- V2: Context-aware (title + article + top-2 similar) → queries
"""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm


# Prompt Templates

PROMPT_V1_TEMPLATE = """
Având titlul și articolul de mai jos, generează 2-3 întrebări 
care surprind ideea principală, fără a copia textul literal.
Întrebările trebuie să fie naturale, reformulate și diverse.

Titlu: {title}
Articol: {content}

Instrucțiuni:
- Fiecare întrebare trebuie să poată fi răspunsă din articol
- Întrebările nu trebuie să fie copiate direct din text
- Variază între întrebări factuale și semantice
- Fiecare întrebare pe o linie nouă, fără numerotare

Întrebări:
"""

PROMPT_V2_TEMPLATE = """
Având titlul, articolul principal și 2 articole contextualizate, 
generează 2-3 întrebări care:
- Surprind tema comună din toate trei articole
- Sunt mai generale, nu specifice unui singur articol
- Ar putea fi răspunse de orice din cele 3 articole
- Nu copiază textul literal

Titlu: {title}
Articolul principal: {content1}

Articole contextualizate (context pentru tema):
---
{content2}
---
{content3}

Instrucțiuni:
- Focalizează pe tema comună
- Generalizează perspectiva
- Fiecare întrebare pe o linie nouă, fără numerotare

Întrebări:
"""


class PromptGenerator:
    
    def __init__(self):
        pass
    
    def generate_prompt_v1(self, title: str, content: str) -> str:

        return PROMPT_V1_TEMPLATE.format(title=title, content=content)
    
    def generate_prompt_v2(self, title: str, content1: str, 
                            content2: str, content3: str) -> str:

        return PROMPT_V2_TEMPLATE.format(title=title, content1=content1,
                                        content2=content2, content3=content3)
    
    def save_prompts(
        self, 
        documents: List[Dict[str, Any]], 
        output_path: str,
        version: str = "V1"
    ) -> None:
        """
        Save prompts to file.
        
        Args:
            documents: List of documents
            output_path: Output file path
            version: Prompt version ("V1" or "V2")
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc=f"Saving {version} prompts"):
                if version == "V1":
                    prompt = self.generate_prompt_v1(
                        doc['title'], 
                        doc['content']
                    )
                elif version == "V2":
                    # For V2, we need similar documents
                    # This is a placeholder - in real scenario, 
                    # we'd fetch similar docs based on embeddings
                    prompt = self.generate_prompt_v2(
                        doc['title'],
                        doc['content'],
                        doc.get('similar_1', 'N/A'),
                        doc.get('similar_2', 'N/A')
                    )
                
                item = {
                    "doc_id": doc.get('doc_id'),
                    "version": version,
                    "prompt": prompt,
                    "title": doc.get('title')
                }
                
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Task 1: Generate LLM Prompts")

    parser.add_argument('--dataset', type=str,
                        choices=['news', 'recipes', 'commoncrawl'],
                        default='news', help='Dataset to use')

    parser.add_argument('--num_docs', type=int, default=100,
                        help='Number of documents to process')

    parser.add_argument('--output_v1', type=str, default='data/prompts_v1.jsonl',
                        help='Output path for V1 prompts')

    parser.add_argument('--output_v2', type=str, default='data/prompts_v2.jsonl',
                        help='Output path for V2 prompts')
    
    args = parser.parse_args()
    
    print(f"\n Task 1: Prompt Generation")
    print(f"   Dataset: {args.dataset}")
    print(f"   Num docs: {args.num_docs}")
    print("-" * 60)
    
    # TODO: Load documents from dataset
    # sample_docs = load_dataset(args.dataset, args.num_docs)
    
    generator = PromptGenerator()
    
    # TODO: Generate and save prompts
    # generator.save_prompts(sample_docs, args.output_v1, version="V1")
    # generator.save_prompts(sample_docs, args.output_v2, version="V2")
    
    print(f"\n Prompts saved!")
    print(f"   V1: {args.output_v1}")
    print(f"   V2: {args.output_v2}")


if __name__ == '__main__':
    main()
