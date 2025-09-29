#!/usr/bin/env python3
"""
Contract Clause Classification Module
=====================================

This module provides functionality to classify contract clauses as "standard" or "non-standard"
by comparing them with template clauses using multiple similarity metrics including semantic analysis.

Features:
- Multi-metric similarity calculation (Cosine, Jaccard, Levenshtein, Jaro-Winkler, Dice, Semantic)
- Contract vs Template clause comparison
- Standard/Non-standard classification with configurable thresholds
- Backend-ready JSON responses
- Support for TN vs TN and WA vs WA comparisons

Author: AI Assistant
"""

import math
import re
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("Sentence Transformers available for semantic similarity")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(" Sentence Transformers not available - semantic similarity will be disabled")


@dataclass
class ClauseComparisonResult:
    """Data class to hold clause comparison results."""
    contract_clause: str
    template_clause: str
    contract_type: str  # TN or WA
    query_name: str
    cosine: float
    jaccard: float
    levenshtein_distance: int
    levenshtein_similarity: float
    jaro_winkler: float
    dice: float
    euclidean_distance: float
    manhattan_distance: int
    overlap: float
    bigram_jaccard: float
    bigram_dice: float
    semantic_similarity: Optional[float] = None
    average_similarity: Optional[float] = None
    weighted_average: Optional[float] = None
    classification: Optional[str] = None  # "standard" or "non_standard"
    confidence_score: Optional[float] = None


class ContractClauseClassifier:
    """
    A classifier for determining if contract clauses are standard or non-standard
    by comparing them with template clauses using multiple similarity metrics.
    """
    
    # Default weights for different similarity metrics
    DEFAULT_WEIGHTS = {
        'cosine': 0.05,
        'jaccard': 0.07,
        'levenshtein_similarity': 0.0,
        'jaro_winkler': 0.05,
        'dice': 0.05,
        'overlap': 0.25,
        'bigram_jaccard': 0.03,
        'bigram_dice': 0.08,
        'semantic_similarity': 0.5 if SENTENCE_TRANSFORMERS_AVAILABLE else 0.0
    }
    
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2', 
                 enable_semantic: bool = True,
                 classification_threshold: float = 0.75,
                 custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the Contract Clause Classifier.
        
        Args:
            model_name: Name of the sentence transformer model
            enable_semantic: Whether to enable semantic similarity calculation
            classification_threshold: Similarity threshold for standard/non-standard classification
            custom_weights: Custom weights for different similarity metrics
        """
        print("Initializing Contract Clause Classifier...")
        
        self.logger = self._setup_logger()
        self.model = None
        self.enable_semantic = enable_semantic and SENTENCE_TRANSFORMERS_AVAILABLE
        self.classification_threshold = classification_threshold
        
        # Set up weights
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)
            self._normalize_weights()
        
        # Initialize semantic model if available and enabled
        if self.enable_semantic:
            try:
                print(f" Loading semantic model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                print("Semantic model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load sentence transformer: {e}")
                self.enable_semantic = False
        
        print(f"Classification threshold: {classification_threshold}")
        print("Contract Clause Classifier initialized successfully!")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up logger for the class."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def _validate_input(self, str1: str, str2: str) -> Tuple[str, str]:
        """Validate and preprocess input strings."""
        if not isinstance(str1, str) or not isinstance(str2, str):
            raise TypeError("Both inputs must be strings")
        
        if not str1.strip() and not str2.strip():
            raise ValueError("Both strings cannot be empty")
        
        return str1.strip(), str2.strip()
    
    @lru_cache(maxsize=128)
    def cosine_similarity(self, str1: str, str2: str) -> float:
        """Calculate cosine similarity between two strings using word vectors."""
        try:
            words1 = re.findall(r'\w+', str1.lower())
            words2 = re.findall(r'\w+', str2.lower())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            all_words = set(words1 + words2)
            vec1 = [words1.count(word) for word in all_words]
            vec2 = [words2.count(word) for word in all_words]
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        
        except Exception as e:
            self.logger.error(f"Error in cosine_similarity: {e}")
            return 0.0
    
    @lru_cache(maxsize=128)
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaccard similarity using word sets."""
        try:
            words1 = set(re.findall(r'\w+', str1.lower()))
            words2 = set(re.findall(r'\w+', str2.lower()))
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error in jaccard_similarity: {e}")
            return 0.0
    
    @lru_cache(maxsize=128)
    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        try:
            if str1 == str2:
                return 0
            
            m, n = len(str1), len(str2)
            
            if m > n:
                str1, str2 = str2, str1
                m, n = n, m
            
            current_row = list(range(m + 1))
            
            for i in range(1, n + 1):
                previous_row, current_row = current_row, [i] + [0] * m
                for j in range(1, m + 1):
                    add = previous_row[j] + 1
                    delete = current_row[j - 1] + 1
                    change = previous_row[j - 1]
                    if str1[j - 1] != str2[i - 1]:
                        change += 1
                    current_row[j] = min(add, delete, change)
            
            return current_row[m]
        
        except Exception as e:
            self.logger.error(f"Error in levenshtein_distance: {e}")
            return max(len(str1), len(str2))
    
    def levenshtein_similarity(self, str1: str, str2: str) -> float:
        """Convert Levenshtein distance to similarity score (0-1)."""
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        distance = self.levenshtein_distance(str1, str2)
        return 1 - (distance / max_len)
    
    @lru_cache(maxsize=128)
    def jaro_winkler_similarity(self, str1: str, str2: str, p: float = 0.1) -> float:
        """Calculate Jaro-Winkler similarity."""
        try:
            def jaro_similarity(s1: str, s2: str) -> float:
                if s1 == s2:
                    return 1.0
                
                len1, len2 = len(s1), len(s2)
                if len1 == 0 or len2 == 0:
                    return 0.0
                
                match_window = max(len1, len2) // 2 - 1
                match_window = max(0, match_window)
                
                s1_matches = [False] * len1
                s2_matches = [False] * len2
                
                matches = 0
                transpositions = 0
                
                # Identify matches
                for i in range(len1):
                    start = max(0, i - match_window)
                    end = min(i + match_window + 1, len2)
                    
                    for j in range(start, end):
                        if s2_matches[j] or s1[i] != s2[j]:
                            continue
                        s1_matches[i] = s2_matches[j] = True
                        matches += 1
                        break
                
                if matches == 0:
                    return 0.0
                
                # Count transpositions
                k = 0
                for i in range(len1):
                    if not s1_matches[i]:
                        continue
                    while not s2_matches[k]:
                        k += 1
                    if s1[i] != s2[k]:
                        transpositions += 1
                    k += 1
                
                jaro = (matches/len1 + matches/len2 + (matches-transpositions/2)/matches) / 3
                return jaro
            
            jaro = jaro_similarity(str1, str2)
            
            # Calculate common prefix length (up to 4 characters)
            prefix = 0
            for i in range(min(len(str1), len(str2), 4)):
                if str1[i] == str2[i]:
                    prefix += 1
                else:
                    break
            
            return jaro + (prefix * p * (1 - jaro))
        
        except Exception as e:
            self.logger.error(f"Error in jaro_winkler_similarity: {e}")
            return 0.0
    
    @lru_cache(maxsize=128)
    def dice_coefficient(self, str1: str, str2: str) -> float:
        """Calculate Dice coefficient using word sets."""
        try:
            words1 = set(re.findall(r'\w+', str1.lower()))
            words2 = set(re.findall(r'\w+', str2.lower()))
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1.intersection(words2))
            total_words = len(words1) + len(words2)
            
            return 2 * intersection / total_words if total_words > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error in dice_coefficient: {e}")
            return 0.0
    
    @lru_cache(maxsize=128)
    def euclidean_distance(self, str1: str, str2: str) -> float:
        """Calculate Euclidean distance using character frequency vectors."""
        try:
            all_chars = set(str1.lower() + str2.lower())
            freq1 = Counter(str1.lower())
            freq2 = Counter(str2.lower())
            
            distance = math.sqrt(sum((freq1[char] - freq2[char]) ** 2 for char in all_chars))
            return distance
        
        except Exception as e:
            self.logger.error(f"Error in euclidean_distance: {e}")
            return float('inf')
    
    @lru_cache(maxsize=128)
    def manhattan_distance(self, str1: str, str2: str) -> int:
        """Calculate Manhattan distance using character frequency vectors."""
        try:
            all_chars = set(str1.lower() + str2.lower())
            freq1 = Counter(str1.lower())
            freq2 = Counter(str2.lower())
            
            distance = sum(abs(freq1[char] - freq2[char]) for char in all_chars)
            return distance
        
        except Exception as e:
            self.logger.error(f"Error in manhattan_distance: {e}")
            return len(str1) + len(str2)
    
    @lru_cache(maxsize=128)
    def overlap_coefficient(self, str1: str, str2: str) -> float:
        """Calculate Overlap coefficient using word sets."""
        try:
            words1 = set(re.findall(r'\w+', str1.lower()))
            words2 = set(re.findall(r'\w+', str2.lower()))
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1.intersection(words2))
            min_size = min(len(words1), len(words2))
            
            return intersection / min_size if min_size > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error in overlap_coefficient: {e}")
            return 0.0
    
    @staticmethod
    def _get_ngrams(text: str, n: int) -> List[str]:
        """Get n-grams from text."""
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    @lru_cache(maxsize=128)
    def ngram_jaccard(self, str1: str, str2: str, n: int = 2) -> float:
        """Calculate Jaccard similarity using character n-grams."""
        try:
            ngrams1 = set(self._get_ngrams(str1.lower(), n))
            ngrams2 = set(self._get_ngrams(str2.lower(), n))
            
            if not ngrams1 and not ngrams2:
                return 1.0
            
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error in ngram_jaccard: {e}")
            return 0.0
    
    @lru_cache(maxsize=128)
    def ngram_dice(self, str1: str, str2: str, n: int = 2) -> float:
        """Calculate Dice coefficient using character n-grams."""
        try:
            ngrams1 = set(self._get_ngrams(str1.lower(), n))
            ngrams2 = set(self._get_ngrams(str2.lower(), n))
            
            if not ngrams1 and not ngrams2:
                return 1.0
            
            intersection = len(ngrams1.intersection(ngrams2))
            total_ngrams = len(ngrams1) + len(ngrams2)
            
            return 2 * intersection / total_ngrams if total_ngrams > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error in ngram_dice: {e}")
            return 0.0
    
    def semantic_similarity(self, str1: str, str2: str) -> Optional[float]:
        """Calculate semantic similarity using sentence transformers."""
        if not self.enable_semantic or not self.model:
            return None
        
        try:
            # Handle empty strings
            if not str1.strip() and not str2.strip():
                return 1.0
            if not str1.strip() or not str2.strip():
                return 0.0
            
            # Get embeddings
            embeddings = self.model.encode([str1, str2])
            
            # Calculate cosine similarity
            embedding1, embedding2 = embeddings[0], embeddings[1]
            
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1) / 2))
        
        except Exception as e:
            self.logger.error(f"Error in semantic_similarity: {e}")
            return None
    
    def compare_clauses(self, contract_clause: str, template_clause: str, 
                       contract_type: str, query_name: str) -> ClauseComparisonResult:
        """
        Compare a contract clause with a template clause and calculate all similarity metrics.
        
        Args:
            contract_clause: The clause from the contract
            template_clause: The clause from the template
            contract_type: Type of contract (TN or WA)
            query_name: Name of the query/attribute being compared
            
        Returns:
            ClauseComparisonResult object containing all calculated similarities
        """
        try:
            contract_clause, template_clause = self._validate_input(contract_clause, template_clause)
            
            print(f" Comparing {contract_type} {query_name} clauses...")
            
            # Calculate all similarities
            cosine = self.cosine_similarity(contract_clause, template_clause)
            jaccard = self.jaccard_similarity(contract_clause, template_clause)
            levenshtein_dist = self.levenshtein_distance(contract_clause, template_clause)
            levenshtein_sim = self.levenshtein_similarity(contract_clause, template_clause)
            jaro_winkler = self.jaro_winkler_similarity(contract_clause, template_clause)
            dice = self.dice_coefficient(contract_clause, template_clause)
            euclidean_dist = self.euclidean_distance(contract_clause, template_clause)
            manhattan_dist = self.manhattan_distance(contract_clause, template_clause)
            overlap = self.overlap_coefficient(contract_clause, template_clause)
            bigram_jaccard = self.ngram_jaccard(contract_clause, template_clause, 2)
            bigram_dice = self.ngram_dice(contract_clause, template_clause, 2)
            semantic_sim = self.semantic_similarity(contract_clause, template_clause)
            
            # Create result object
            result = ClauseComparisonResult(
                contract_clause=contract_clause,
                template_clause=template_clause,
                contract_type=contract_type,
                query_name=query_name,
                cosine=cosine,
                jaccard=jaccard,
                levenshtein_distance=levenshtein_dist,
                levenshtein_similarity=levenshtein_sim,
                jaro_winkler=jaro_winkler,
                dice=dice,
                euclidean_distance=euclidean_dist,
                manhattan_distance=manhattan_dist,
                overlap=overlap,
                bigram_jaccard=bigram_jaccard,
                bigram_dice=bigram_dice,
                semantic_similarity=semantic_sim
            )
            
            # Calculate averages
            result.average_similarity = self._calculate_average_similarity(result)
            result.weighted_average = self._calculate_weighted_average(result)
            
            # Classify as standard or non-standard
            result.classification = self._classify_clause(result.weighted_average)
            result.confidence_score = result.weighted_average
            
            print(f"Comparison complete - Classification: {result.classification} (Score: {result.weighted_average:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in compare_clauses: {e}")
            raise
    
    def _calculate_average_similarity(self, result: ClauseComparisonResult) -> float:
        """Calculate simple average of all similarity scores."""
        similarities = [
            result.cosine, result.jaccard, result.levenshtein_similarity,
            result.jaro_winkler, result.dice, result.overlap,
            result.bigram_jaccard, result.bigram_dice
        ]
        
        if result.semantic_similarity is not None:
            similarities.append(result.semantic_similarity)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_weighted_average(self, result: ClauseComparisonResult) -> float:
        """Calculate weighted average of similarity scores."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        similarity_values = {
            'cosine': result.cosine,
            'jaccard': result.jaccard,
            'levenshtein_similarity': result.levenshtein_similarity,
            'jaro_winkler': result.jaro_winkler,
            'dice': result.dice,
            'overlap': result.overlap,
            'bigram_jaccard': result.bigram_jaccard,
            'bigram_dice': result.bigram_dice,
            'semantic_similarity': result.semantic_similarity
        }
        
        for metric, value in similarity_values.items():
            if value is not None and metric in self.weights:
                weighted_sum += value * self.weights[metric]
                total_weight += self.weights[metric]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _classify_clause(self, similarity_score: float) -> str:
        """Classify clause as standard or non-standard based on similarity score."""
        return "standard" if similarity_score >= self.classification_threshold else "non_standard"
    
    def batch_compare_clauses(self, contract_clauses: List[Dict], 
                             template_clauses: List[Dict]) -> List[ClauseComparisonResult]:
        """
        Compare multiple contract clauses with their corresponding template clauses.
        
        Args:
            contract_clauses: List of contract clause dictionaries with keys: 'text', 'type', 'query'
            template_clauses: List of template clause dictionaries with keys: 'text', 'type', 'query'
            
        Returns:
            List of ClauseComparisonResult objects
        """
        print(f" Starting batch comparison of {len(contract_clauses)} contract clauses...")
        
        results = []
        
        for contract_clause in contract_clauses:
            # Find matching template clause (same type and query)
            matching_template = None
            for template_clause in template_clauses:
                if (contract_clause['type'] == template_clause['type'] and 
                    contract_clause['query'] == template_clause['query']):
                    matching_template = template_clause
                    break
            
            if matching_template:
                try:
                    result = self.compare_clauses(
                        contract_clause=contract_clause['text'],
                        template_clause=matching_template['text'],
                        contract_type=contract_clause['type'],
                        query_name=contract_clause['query']
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to compare clause for {contract_clause['query']}: {e}")
            else:
                self.logger.warning(f"No matching template found for {contract_clause['type']} {contract_clause['query']}")
        
        print(f"Batch comparison complete: {len(results)} comparisons successful")
        return results
    
    def get_classification_summary(self, results: List[ClauseComparisonResult]) -> Dict:
        """
        Generate a summary of classification results.
        
        Args:
            results: List of ClauseComparisonResult objects
            
        Returns:
            Dictionary containing classification summary
        """
        if not results:
            return {"error": "No results to summarize"}
        
        total_clauses = len(results)
        standard_count = len([r for r in results if r.classification == "standard"])
        non_standard_count = total_clauses - standard_count
        
        avg_similarity = sum(r.weighted_average for r in results) / total_clauses
        
        # Group by contract type
        tn_results = [r for r in results if r.contract_type == "TN"]
        wa_results = [r for r in results if r.contract_type == "WA"]
        
        summary = {
            "total_clauses": total_clauses,
            "standard_clauses": standard_count,
            "non_standard_clauses": non_standard_count,
            "standard_percentage": (standard_count / total_clauses) * 100,
            "average_similarity": avg_similarity,
            "classification_threshold": self.classification_threshold,
            "by_contract_type": {
                "TN": {
                    "total": len(tn_results),
                    "standard": len([r for r in tn_results if r.classification == "standard"]),
                    "non_standard": len([r for r in tn_results if r.classification == "non_standard"]),
                    "avg_similarity": sum(r.weighted_average for r in tn_results) / len(tn_results) if tn_results else 0
                },
                "WA": {
                    "total": len(wa_results),
                    "standard": len([r for r in wa_results if r.classification == "standard"]),
                    "non_standard": len([r for r in wa_results if r.classification == "non_standard"]),
                    "avg_similarity": sum(r.weighted_average for r in wa_results) / len(wa_results) if wa_results else 0
                }
            },
            "detailed_results": [asdict(result) for result in results]
        }
        
        return summary


if __name__ == "__main__":
    # Test the classifier with sample data
    print("Testing Contract Clause Classifier...")
    
    classifier = ContractClauseClassifier(classification_threshold=0.75)
    
    # Sample contract and template clauses
    contract_clause = "Provider shall submit claims within 90 days of service date"
    template_clause = "Provider must file claims within 120 days of service date"
    
    # Compare clauses
    result = classifier.compare_clauses(
        contract_clause=contract_clause,
        template_clause=template_clause,
        contract_type="TN",
        query_name="Timely Filing"
    )
    
    print(f"\nComparison Result:")
    print(f"Classification: {result.classification}")
    print(f"Weighted Average Similarity: {result.weighted_average:.3f}")
    print(f"Semantic Similarity: {result.semantic_similarity:.3f}" if result.semantic_similarity else "Semantic similarity not available")
