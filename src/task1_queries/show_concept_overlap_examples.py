import sys
from pathlib import Path
"""
Show examples of concept overlap analysis for multi-document scenarios.
"""

import json
import re
from collections import Counter
from pathlib import Path

class ConceptExtractor:
    def __init__(self):
        self.stopwords = {
            'și', 'din', 'cu', 'să', 'pe', 'la', 'de', 'el', 'ea', 'este', 'are', 
            'sunt', 'în', 'că', 'ce', 'care', 'sau', 'nu', 'un', 'o', 'for', 'the', 
            'and', 'is', 'are', 'was', 'were', 'a', 'an', 'be', 'am', 'fi', 's-a'
        }
    
    def extract_concepts(self, text, max_concepts=10):
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [
            w for w in words 
            if len(w) > 3 and w not in self.stopwords and w.isalpha()
        ]
        freq = Counter(concepts)
        return [word for word, _ in freq.most_common(max_concepts)]


QUERIES_PATH = "data/queries/queries_gemma3.jsonl"


def analyze_concept_overlap():
    """Show concept overlap for related documents."""
    
    extractor = ConceptExtractor()
    
    print("\n" + "="*80)
    print("CONCEPT OVERLAP ANALYSIS - MULTI-DOCUMENT SCENARIOS")
    print("="*80)
    
    # Load some query records
    records_by_type = {}
    with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                doc_type = record.get("type")
                
                if doc_type not in records_by_type:
                    records_by_type[doc_type] = []
                
                if len(records_by_type[doc_type]) < 3:  # Take 3 samples per type
                    records_by_type[doc_type].append(record)
                
                if all(len(v) >= 3 for v in records_by_type.values()):
                    break
            except:
                pass
    
    # Analyze overlap for each type
    for doc_type, records in sorted(records_by_type.items()):
        print(f"\n{'='*80}")
        print(f"📚 {doc_type.upper()} - Analyzing {len(records)} related documents")
        print(f"{'='*80}")
        
        # Extract concepts from all documents
        all_doc_concepts = []
        for i, record in enumerate(records):
            title = record.get("title", "")
            doc_text = title[:100]  # Use title as proxy
            concepts = extractor.extract_concepts(doc_text, max_concepts=15)
            all_doc_concepts.append(set(concepts))
            print(f"\n  Doc {i+1}: \"{title[:60]}...\"")
            print(f"    Concepts: {', '.join(concepts[:8])}")
        
        # Analyze overlap
        print(f"\n  Concept Overlap Analysis:")
        
        # Shared in ALL documents
        if all_doc_concepts:
            shared_all = set.intersection(*all_doc_concepts) if len(all_doc_concepts) > 1 else all_doc_concepts[0]
            print(f"    • Concepts in ALL {len(records)} docs: {', '.join(shared_all) if shared_all else '(none)'}")
        
        # Shared in ANY document
        if all_doc_concepts:
            shared_any = set.union(*all_doc_concepts)
            print(f"    • Concepts in ANY doc: {len(shared_any)} unique concepts")
        
        # Show query samples with metrics
        print(f"\n  Sample Queries & Metrics:")
        for i, record in enumerate(records):
            queries = record.get("queries", [])[:2]  # First 2 queries
            metrics = record.get("metrics", {})
            
            print(f"\n    Doc {i+1}:")
            for j, query in enumerate(queries):
                quality = metrics.get("quality_scores", [0])[j] if j < len(metrics.get("quality_scores", [])) else 0
                print(f"      Q{j+1}: \"{query[:50]}...\"")
                print(f"          Quality: {quality:.3f}")
            
            print(f"      Avg Quality: {metrics.get('avg_quality', 0):.3f}")
            print(f"      Diversity: {metrics.get('diversity', 0):.3f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*80}")
    print("""
✓ CONCEPT OVERLAP:
  • Concepts appearing in ALL related documents = highly relevant concepts
  • These concepts should be targeted in query-document matching
  • For training triplets: overlapping concepts ensure relevance

✓ QUALITY METRICS MEANING:
  • quality_score (0-1): How well query matches document content
    - 0.60+: High quality (good concept coverage)
    - 0.50-0.60: Medium quality (acceptable for training)
    - <0.50: Low quality (need improvement)
  
  • diversity (0-1): How different queries are from each other
    - 1.0: Perfect diversity (each query is unique)
    - 0.5: Half diversity (some query overlap)
    - <0.3: Low diversity (queries are repetitive - problem)
  
  • avg_quality per document: Overall quality of query set for that document

✓ USAGE FOR TRAINING:
  • Filter out queries with quality < 0.50 for strict training
  • Keep quality 0.50-0.70 for data augmentation
  • High diversity (>0.5) ensures varied training examples
  • Concept overlap helps with semantic understanding
    """)

if __name__ == "__main__":
    analyze_concept_overlap()
