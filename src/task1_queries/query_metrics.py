"""
Query quality metrics (fast, dependency-free).

Extracted from the old template generator so that any query generator
(templates, LLM, manual) can reuse the same scoring:

1. Query-document concept overlap (fast string-based, no TF-IDF)
2. Query length and triviality scoring
3. Query diversity within a document's query set

Pure stdlib on purpose: these helpers must import on a login node with no
scientific stack installed (e.g. for `--dry-run`).
"""

import re
from typing import List, Dict, Any


def mean(values: List[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    """Population standard deviation, 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5


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
        from collections import Counter

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

        return 1.0 - mean(similarities)

    def score_record(self, queries: List[str], doc: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the full metrics block for one document's queries."""
        if not queries:
            return {}

        quality_scores = [self.quality_score(q, doc) for q in queries]
        return {
            "quality_scores": [float(s) for s in quality_scores],
            "avg_quality": float(mean(quality_scores)),
            "diversity": float(self.diversity_score(queries)),
            "num_queries": len(queries),
        }
