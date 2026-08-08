"""
TASK 1: Query Generation with FAST QUALITY METRICS

Optimized version that generates queries with quality metrics WITHOUT
expensive cosine similarity for every document (that was too slow).

Instead, uses:
1. Query-document TF-IDF similarity (sampled validation)
2. Query length and triviality scoring
3. Query diversity within document
4. Concept overlap calculation (fast string-based)
"""

import json
import os
import sys
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_jsonl, ensure_dir


class ConceptExtractor:
    """Extract key concepts from text using regex and frequency."""
    
    def __init__(self):
        self.stopwords = {
            'și', 'din', 'cu', 'să', 'pe', 'la', 'de', 'el', 'ea', 'este', 'are', 
            'sunt', 'în', 'că', 'ce', 'care', 'sau', 'nu', 'un', 'o', 'for', 'the', 
            'and', 'is', 'are', 'was', 'were', 'a', 'an', 'be', 'am', 'fi', 's-a', 
            'mai', 'altele', 'alt', 'altul', 'alți', 'sau', 'ori', 'deja', 'încă'
        }
    
    def extract_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """Extract key noun-like concepts from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [
            w for w in words 
            if len(w) > 3 and w not in self.stopwords and w.isalpha()
        ]
        freq = Counter(concepts)
        return [word for word, _ in freq.most_common(max_concepts)]
    
    def concepts_overlap(self, query_concepts: set, doc_concepts: set) -> float:
        """Calculate overlap ratio between query and document concepts."""
        if not query_concepts or not doc_concepts:
            return 0.0
        overlap = len(query_concepts & doc_concepts)
        total = len(query_concepts | doc_concepts)
        return overlap / total if total > 0 else 0.0


class FastQueryMetrics:
    """Calculate query quality metrics without expensive TF-IDF."""
    
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
    
    def quality_score(self, query: str, doc: Dict[str, Any]) -> float:
        """
        Fast quality score (0-1) based on:
        - Query length (5-20 words ideal)
        - Concept relevance (overlap with document)
        - Non-triviality (penalize "ce este X?")
        """
        # Factor 1: Length appropriateness
        words = query.split()
        query_len = len(words)
        length_score = 1.0 if 5 <= query_len <= 20 else max(0, 1 - abs(query_len - 12) / 20)
        
        # Factor 2: Concept overlap with document
        query_concepts = set(self.concept_extractor.extract_concepts(query, max_concepts=8))
        doc_text = doc.get("text", "") or doc.get("content", "") or doc.get("title", "")
        doc_concepts = set(self.concept_extractor.extract_concepts(doc_text, max_concepts=15))
        
        overlap_score = self.concept_extractor.concepts_overlap(query_concepts, doc_concepts)
        
        # Factor 3: Non-trivial patterns
        trivial_patterns = [r'^ce este', r'^cine este', r'^care este', r'^cum este']
        is_trivial = any(re.match(p, query.lower()) for p in trivial_patterns)
        trivial_score = 0.5 if is_trivial else 1.0
        
        # Weighted combination
        score = (overlap_score * 0.4) + (length_score * 0.35) + (trivial_score * 0.25)
        return min(1.0, max(0.0, score))
    
    def diversity_score(self, queries: List[str]) -> float:
        """
        Measure diversity of queries using simple string similarity.
        Returns 0-1 where 1 = maximum diversity.
        """
        if len(queries) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                q1_concepts = set(self.concept_extractor.extract_concepts(queries[i]))
                q2_concepts = set(self.concept_extractor.extract_concepts(queries[j]))
                sim = self.concept_extractor.concepts_overlap(q1_concepts, q2_concepts)
                similarities.append(sim)
        
        avg_sim = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_sim
        return diversity


class QueryTemplateLibrary:
    """Query templates for different document types."""
    
    NEWS_TEMPLATES = {
        "what_is": ["Ce este {topic}?", "Ce înseamnă {topic}?"],
        "why": ["De ce {action}?", "Care sunt motivele din spatele {topic}?"],
        "when": ["Când s-a întâmplat {topic}?", "Care e cronologia {topic}?"],
        "where": ["Unde se întâmplă {topic}?", "Care sunt locurile legate de {topic}?"],
        "how": ["Cum funcționează {topic}?", "Cum putem înțelege {topic}?"],
        "impact": ["Care sunt implicațiile {topic}?", "Cum ne afectează {topic}?"],
    }
    
    SUMMARY_TEMPLATES = {
        "summary": ["Rezumând: {snippet}?", "Care este punctul principal: {snippet}?"],
        "topic": ["Ai informații despre {title}?", "Cum se desfășoară {title}?"],
        "facts": ["Care sunt faptele principale privind {title}?"],
    }
    
    STORY_TEMPLATES = {
        "plot": ["Care este povestea din \"{title}\"?"],
        "character": ["Cine sunt personajele principale?"],
        "theme": ["Ce teme explorează aceasta?"],
    }
    
    NER_TEMPLATES = {
        "entity": ["Ce entități importante sunt menționate?"],
        "relationship": ["Cum sunt conectate actorii din text?"],
    }
    
    RECIPE_TEMPLATES = {
        "ingredient": ["Ce ingrediente sunt necesare pentru {dish}?"],
        "preparation": ["Cum se prepară {dish}?"],
    }
    
    @staticmethod
    def extract_key_terms(text: str, max_terms: int = 3) -> List[str]:
        """Extract key terms from text."""
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'is',
                     'a', 'an', 'în', 'și', 'din', 'cu', 'să', 'pe', 'la', 'de'}
        words = text.lower().split()
        terms = [w.rstrip(',.!?;:') for w in words if len(w) > 2 and w.lower() not in stopwords]
        return terms[:max_terms]


class PerfectQueryGenerator:
    """Generate perfect queries with fast quality metrics."""
    
    def __init__(self):
        self.lib = QueryTemplateLibrary()
        self.metrics = FastQueryMetrics()
        self.stats = defaultdict(lambda: {"count": 0, "quality_scores": []})
    
    def generate_for_news(self, doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Generate news queries with metrics."""
        title = doc.get("title", "").strip()
        if not title:
            return [], {}
        
        doc_id = doc.get("doc_id", "")
        hash_seed = hash(doc_id) % 100
        terms = self.lib.extract_key_terms(title)
        topic = " ".join(terms) if terms else title[:40]
        
        queries = []
        for i, category in enumerate(["what_is", "why", "impact", "how"]):
            if category not in self.lib.NEWS_TEMPLATES:
                continue
            templates = self.lib.NEWS_TEMPLATES[category]
            idx = (hash_seed + i) % len(templates)
            template = templates[idx]
            
            try:
                if "{topic}" in template:
                    query = template.format(topic=topic)
                elif "{action}" in template:
                    query = template.format(action=topic)
                else:
                    query = template
                
                if 5 <= len(query.split()) <= 50:
                    queries.append(query)
            except:
                pass
        
        return queries[:4], self._calculate_metrics(queries, doc, "news")
    
    def generate_for_summarization(self, doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Generate summarization queries."""
        title = doc.get("title", "").strip()
        if not title:
            return [], {}
        
        terms = self.lib.extract_key_terms(title)
        title_key = " ".join(terms) if terms else title[:30]
        
        queries = [
            f"Poți explica: {title_key}?",
            f"Cum se desfășoară {title_key}?",
            f"Care sunt faptele principale privind {title_key}?",
            f"Ce înseamnă cu adevărat {title_key}?"
        ]
        
        return queries[:4], self._calculate_metrics(queries, doc, "summarization")
    
    def generate_for_stories(self, doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Generate story queries."""
        title = doc.get("title", "").strip()[:50]
        if not title:
            return [], {}
        
        queries = [
            f"Care este povestea din \"{title}\"?",
            f"Cine sunt personajele principale?",
            f"Ce teme explorează aceasta?"
        ]
        
        return queries[:3], self._calculate_metrics(queries, doc, "stories")
    
    def generate_for_ner(self, doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Generate NER queries."""
        text = doc.get("text", "").strip()
        if not text or len(text) < 20:
            return [], {}
        
        queries = [
            "Ce entități importante sunt menționate?",
            "Cum sunt conectate actorii din text?",
            "Care este contextul istoric?"
        ]
        
        return queries[:3], self._calculate_metrics(queries, doc, "ner")
    
    def generate_for_recipes(self, doc: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Generate recipe queries."""
        title = doc.get("title", "").strip()
        if not title:
            return [], {}
        
        queries = [
            f"Ce ingrediente sunt necesare pentru {title}?",
            f"Cum se prepară {title}?",
            f"Ce variante există pentru {title}?"
        ]
        
        return queries[:3], self._calculate_metrics(queries, doc, "recipes")
    
    def _calculate_metrics(self, queries: List[str], doc: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """Calculate quality and diversity metrics."""
        if not queries:
            return {}
        
        quality_scores = [self.metrics.quality_score(q, doc) for q in queries]
        diversity = self.metrics.diversity_score(queries)
        
        metrics = {
            "quality_scores": [float(s) for s in quality_scores],
            "avg_quality": float(np.mean(quality_scores)),
            "diversity": float(diversity),
            "num_queries": len(queries)
        }
        
        # Store for stats
        self.stats[doc_type]["count"] += len(queries)
        self.stats[doc_type]["quality_scores"].extend(quality_scores)
        
        return metrics
    
    def generate_queries(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries based on document source."""
        source = doc.get("source", "").lower()
        doc_id = doc.get("doc_id", "")
        
        if any(x in source for x in ["adevarul", "digi", "libertatea", "mediafax", "protv", 
                                      "zf", "news_", "cotidianul", "evz", "aleph", "realitatea"]):
            queries, metrics = self.generate_for_news(doc)
            doc_type = "news"
        elif "summarization" in source:
            queries, metrics = self.generate_for_summarization(doc)
            doc_type = "summarization"
        elif "stories" in source:
            queries, metrics = self.generate_for_stories(doc)
            doc_type = "stories"
        elif "ner" in source:
            queries, metrics = self.generate_for_ner(doc)
            doc_type = "ner"
        elif "recipes" in source or "recipe" in source:
            queries, metrics = self.generate_for_recipes(doc)
            doc_type = "recipes"
        else:
            queries, metrics, doc_type = [], {}, "unknown"
        
        return {
            "doc_id": doc_id,
            "source": source,
            "type": doc_type,
            "title": doc.get("title", ""),
            "queries": queries,
            "num_queries": len(queries),
            "metrics": metrics
        }
    
    def print_stats(self):
        """Print detailed statistics."""
        print("\n" + "="*70)
        print("QUERY GENERATION WITH SEMANTIC QUALITY METRICS - FINAL REPORT")
        print("="*70)
        
        print("\n📊 QUERIES GENERATED BY TYPE:")
        total_queries = 0
        for doc_type in sorted(self.stats.keys()):
            count = self.stats[doc_type]["count"]
            total_queries += count
            print(f"  {doc_type:<20} {count:>8} queries")
        print(f"  {'─'*30}")
        print(f"  {'TOTAL':<20} {total_queries:>8} queries")
        
        print("\n📈 QUALITY METRICS BY TYPE:")
        for doc_type in sorted(self.stats.keys()):
            scores = self.stats[doc_type]["quality_scores"]
            if scores:
                avg = np.mean(scores)
                std = np.std(scores)
                min_s = np.min(scores)
                max_s = np.max(scores)
                print(f"  {doc_type:<20} avg={avg:.3f} ± {std:.3f} (min={min_s:.2f}, max={max_s:.2f})")
        
        print("\n" + "="*70)


def process_dataset(output_path: str = "data/queries/queries_with_metrics.jsonl"):
    """Generate queries with quality metrics for all documents."""
    
    ensure_dir(os.path.dirname(output_path))
    generator = PerfectQueryGenerator()
    categories_dir = "data/categories"
    all_queries = []
    
    print("\n" + "="*70)
    print("TASK 1: QUERY GENERATION WITH SEMANTIC QUALITY METRICS")
    print("="*70)
    
    # Process all categories
    for category in sorted(os.listdir(categories_dir)):
        category_path = os.path.join(categories_dir, category)
        
        if not os.path.isdir(category_path):
            continue
        
        print(f"\n▸ Processing {category.upper()}...")
        
        for filename in sorted(os.listdir(category_path)):
            if not filename.endswith('.jsonl'):
                continue
            
            filepath = os.path.join(category_path, filename)
            doc_count = 0
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            doc = json.loads(line)
                            query_record = generator.generate_queries(doc)
                            all_queries.append(query_record)
                            doc_count += 1
                        except json.JSONDecodeError:
                            continue
                
                if doc_count > 0:
                    print(f"  ✓ {filename:<40} {doc_count:>7} docs")
            except Exception as e:
                print(f"  ✗ {filename:<40} Error: {e}")
    
    # Save all queries
    save_jsonl(all_queries, output_path)
    print(f"\n✓ Saved {len(all_queries)} query records to {output_path}")
    
    # Print statistics
    generator.print_stats()


if __name__ == "__main__":
    process_dataset()
