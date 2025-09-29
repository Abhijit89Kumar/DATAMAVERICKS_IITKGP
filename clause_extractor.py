#!/usr/bin/env python3
"""
Clause Extractor Module - Intelligent Text Processing and Search
===============================================================

This module contains classes for:
- Text preprocessing and cleaning
- Hierarchical document chunking based on markdown structure
- Hybrid semantic + keyword search capabilities

Features:
- Hierarchical document chunking
- Semantic similarity search
- TF-IDF keyword matching
- Contract clause extraction and analysis

Author: AI Assistant
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core imports
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    print(" Downloading NLTK data...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data downloaded successfully")
        NLTK_AVAILABLE = True
    except Exception as e:
        print(f" NLTK data download failed: {e}")
        NLTK_AVAILABLE = False
except ImportError:
    print(" NLTK not available - using basic text preprocessing")
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Handles text preprocessing including stopword removal and lemmatization.
    """
    
    def __init__(self):
        print(" Initializing text preprocessor...")
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                print("Text preprocessor initialized with NLTK")
            except Exception as e:
                print(f" Text preprocessor initialization warning: {e}")
                # Fallback to basic preprocessing
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
                self.lemmatizer = None
        else:
            # Fallback to basic preprocessing
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
            print("Text preprocessor initialized with basic settings")
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text: lowercase, tokenize, remove stopwords, lemmatize.
        """
        try:
            # Lowercase
            text = text.lower()
            
            if NLTK_AVAILABLE and self.lemmatizer:
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords and non-alphabetic tokens, then lemmatize
                processed_tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token.isalpha() and token not in self.stop_words
                ]
            else:
                # Basic tokenization fallback
                tokens = text.split()
                processed_tokens = [
                    token
                    for token in tokens
                    if token.isalpha() and token not in self.stop_words
                ]
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f" Text preprocessing failed: {e}")
            # Return basic cleaned text as fallback
            return ' '.join([word.lower() for word in text.split() if word.isalpha()])


class MarkdownHierarchicalChunker:
    """
    Chunks markdown documents based on hierarchical section structure.
    """
    
    def __init__(self):
        print(" Initializing hierarchical chunker...")
        self.chunks = []
        print("Hierarchical chunker initialized")
        
    def parse_markdown(self, md_text: str) -> List[Dict]:
        """
        Parse markdown text and extract hierarchical sections.
        """
        print(" Parsing markdown structure...")
        lines = md_text.split('\n')
        sections = []
        current_section = None
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Match markdown headers (## Header)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line_stripped)
            
            # Match list items with section numbers (- 3.1 Title)
            list_match = re.match(r'^[-]\s+([\d\.]+)\s+(.+?)\.?\s(.*)$', line_stripped)
            
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Check if header has a section number (e.g., "## 3 Title" or "## ARTICLE III")
                section_num_match = re.match(r'^([\d\.]+|ARTICLE\s+[IVX]+)\s+(.+)$', title, re.IGNORECASE)
                
                if section_num_match:
                    section_num = section_num_match.group(1)
                    section_title = section_num_match.group(2)
                    is_numbered = True
                else:
                    # Plain header like "## MEDICAID"
                    section_num = f"header_{len(sections)}"
                    section_title = title
                    is_numbered = False
                
                current_section = {
                    'type': 'header',
                    'level': level,
                    'section_num': section_num,
                    'title': section_title,
                    'full_title': title,
                    'content': '',
                    'is_numbered': is_numbered,
                    'children': [],
                    'line_num': line_num
                }
                sections.append(current_section)
                
            elif list_match:
                section_num = list_match.group(1)
                section_title = list_match.group(2).strip()
                content = list_match.group(3).strip()
                
                section = {
                    'type': 'list',
                    'section_num': section_num,
                    'title': section_title,
                    'full_title': f"{section_num} {section_title}",
                    'content': content,
                    'is_numbered': True,
                    'children': [],
                    'line_num': line_num
                }
                sections.append(section)
                current_section = section
                
            elif current_section is not None:
                # Add content to current section (skip empty lines)
                if not re.match(r'^[-*]\s+[\d\.]+', line_stripped):
                    if current_section['content']:
                        current_section['content'] += ' ' + line_stripped
                    else:
                        current_section['content'] = line_stripped
        
        print(f"Parsed {len(sections)} sections from markdown")
        return sections
    
    def create_hierarchical_chunks(self, sections: List[Dict]) -> List[Dict]:
        """
        Create chunks from sections.
        Strategy:
        1. Every header (numbered or unnumbered) becomes its own chunk
        2. Every 2-level list item (like "3.1", "2.1") becomes its own chunk with subsections
        3. Subsections (like "3.1.1") are included in their parent chunk only
        """
        print(" Creating hierarchical chunks...")
        chunks = []
        processed = set()
        
        for i, section in enumerate(sections):
            if i in processed:
                continue
            
            # Handle ALL headers (numbered and unnumbered) - each becomes own chunk
            if section['type'] == 'header':
                chunk_text = f"{'#' * section['level']} {section['full_title']}\n\n{section['content']}"
                
                chunks.append({
                    'section_num': section['section_num'],
                    'title': section['full_title'],
                    'text': chunk_text.strip(),
                    'num_subsections': 0,
                    'type': 'header'
                })
                processed.add(i)
                continue
            
            # Handle list items
            if section['type'] == 'list' and section['is_numbered']:
                section_num = section['section_num']
                parts = section_num.split('.')
                
                # Only create chunks for 2-level sections (e.g., "3.1", "2.1")
                # Skip deeper levels (e.g., "3.1.1") - they'll be included in parent
                if len(parts) != 2:
                    continue
                
                # Start building the chunk with this section
                chunk_text = f"{section['full_title']}\n\n{section['content']}\n"
                subsection_count = 0
                
                # Find all subsections (3.1.1, 3.1.2, etc.)
                j = i + 1
                while j < len(sections):
                    next_sec = sections[j]
                    
                    if next_sec['type'] != 'list' or not next_sec.get('is_numbered'):
                        j += 1
                        continue
                    
                    next_num = next_sec['section_num']
                    next_parts = next_num.split('.')
                    
                    # Check if it's a subsection of current section
                    # e.g., 3.1.1, 3.1.2 are subsections of 3.1
                    if len(next_parts) > 2 and next_parts[0] == parts[0] and next_parts[1] == parts[1]:
                        processed.add(j)
                        chunk_text += f"\n*{next_sec['full_title']}*\n{next_sec['content']}\n"
                        subsection_count += 1
                    # Stop if we've moved to a different 2-level section
                    elif len(next_parts) == 2:
                        break
                    
                    j += 1
                
                chunks.append({
                    'section_num': section['section_num'],
                    'title': section['full_title'],
                    'text': chunk_text.strip(),
                    'num_subsections': subsection_count,
                    'type': 'list'
                })
                processed.add(i)
        
        print(f"Created {len(chunks)} hierarchical chunks")
        return chunks
    
    def chunk_markdown_text(self, md_text: str) -> List[Dict]:
        """
        Process markdown text and create hierarchical chunks.
        """
        print(" Processing markdown text for chunking...")
        sections = self.parse_markdown(md_text)
        self.chunks = self.create_hierarchical_chunks(sections)
        
        # Add chunk statistics
        for i, chunk in enumerate(self.chunks):
            chunk['chunk_id'] = i
            chunk['word_count'] = len(chunk['text'].split())
            chunk['char_count'] = len(chunk['text'])
        
        print(f"Chunking complete: {len(self.chunks)} chunks created")
        return self.chunks


class HybridSearcher:
    """
    Performs hybrid search combining semantic similarity and TF-IDF keyword matching.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', 
                 semantic_weight: float = 1, tfidf_weight: float = 0):
        """
        Initialize with a sentence transformer model and weights.
        """
        print(f" Loading semantic search model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print("Semantic model loaded successfully")
        except Exception as e:
            print(f" Failed to load model: {e}")
            raise
            
        self.preprocessor = TextPreprocessor()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        self.semantic_weight = semantic_weight
        self.tfidf_weight = tfidf_weight
        
        self.chunks = None
        self.chunk_embeddings = None
        self.tfidf_matrix = None
        
        print(f"Search weights - Semantic: {semantic_weight}, TF-IDF: {tfidf_weight}")
        print("Hybrid searcher initialized successfully")
        
    def encode_chunks(self, chunks: List[Dict]) -> None:
        """
        Create both semantic embeddings and TF-IDF vectors for all chunks.
        """
        if not chunks:
            print(" Warning: No chunks to encode!")
            return
            
        print(f" Encoding {len(chunks)} chunks...")
        self.chunks = chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Create semantic embeddings
        print(" Creating semantic embeddings...")
        try:
            self.chunk_embeddings = self.model.encode(
                chunk_texts, 
                show_progress_bar=True,
                batch_size=32
            )
            print("Semantic embeddings created")
        except Exception as e:
            print(f" Failed to create semantic embeddings: {e}")
            raise
        
        # Preprocess texts for TF-IDF
        print(" Preprocessing texts for TF-IDF...")
        try:
            preprocessed_chunks = [self.preprocessor.preprocess(text) for text in chunk_texts]
            
            # Create TF-IDF vectors
            print("Creating TF-IDF vectors...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_chunks)
            print("TF-IDF vectors created")
        except Exception as e:
            print(f" Failed to create TF-IDF vectors: {e}")
            raise
        
        print("Chunk encoding complete!")
    
    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float, Dict]]:
        """
        Find the most similar chunks using hybrid search (semantic + TF-IDF).
        
        Returns:
            List[Tuple[Dict, float, Dict]]: List of (chunk, combined_score, score_details)
        """
        if self.chunk_embeddings is None or self.tfidf_matrix is None:
            raise ValueError(" Must encode chunks first using encode_chunks()")
        
        print(f"\n Searching for: '{query}'")
        print(f"Searching through {len(self.chunks)} chunks...")
        
        try:
            # Semantic similarity
            print(" Computing semantic similarity...")
            query_embedding = self.model.encode([query])
            semantic_scores = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            
            # TF-IDF similarity
            print(" Computing TF-IDF similarity...")
            preprocessed_query = self.preprocessor.preprocess(query)
            query_tfidf = self.tfidf_vectorizer.transform([preprocessed_query])
            tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            # Normalize semantic scores to [0, 1]
            semantic_scores_norm = (semantic_scores + 1) / 2
            
            # Combine scores
            combined_scores = (self.semantic_weight * semantic_scores_norm + 
                              self.tfidf_weight * tfidf_scores)
            
            # Get top-k
            top_indices = np.argsort(combined_scores)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                score_details = {
                    'semantic': float(semantic_scores[idx]),
                    'semantic_norm': float(semantic_scores_norm[idx]),
                    'tfidf': float(tfidf_scores[idx]),
                    'combined': float(combined_scores[idx]),
                    'rank': rank + 1
                }
                results.append((self.chunks[idx], combined_scores[idx], score_details))
            
            print(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f" Search failed: {e}")
            raise


class ContractClauseExtractor:
    """
    High-level class that combines PDF parsing, chunking, and search for contract analysis.
    """
    
    def __init__(self, semantic_weight: float = 1, tfidf_weight: float = 0,
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the clause extractor.
        """
        print("Initializing Contract Clause Extractor...")
        
        self.chunker = MarkdownHierarchicalChunker()
        self.searcher = HybridSearcher(
            model_name=model_name,
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight
        )
        
        self.chunks = None
        self.is_ready = False
        
        print("Contract Clause Extractor initialized successfully!")
    
    def process_markdown(self, markdown_content: str) -> Dict[str, Any]:
        """
        Process markdown content and prepare for search.
        """
        print(f" Processing markdown content ({len(markdown_content)} characters)")
        
        # Create chunks
        self.chunks = self.chunker.chunk_markdown_text(markdown_content)
        
        if not self.chunks:
            print(" Warning: No chunks created from markdown content!")
            return {"status": "error", "message": "No chunks created"}
        
        # Encode chunks for search
        self.searcher.encode_chunks(self.chunks)
        self.is_ready = True
        
        # Calculate statistics
        stats = {
            'total_chunks': len(self.chunks),
            'avg_words_per_chunk': np.mean([chunk['word_count'] for chunk in self.chunks]),
            'avg_chars_per_chunk': np.mean([chunk['char_count'] for chunk in self.chunks]),
            'total_words': sum([chunk['word_count'] for chunk in self.chunks]),
            'total_chars': sum([chunk['char_count'] for chunk in self.chunks])
        }
        
        return {
            "status": "success",
            "chunks_created": len(self.chunks),
            "statistics": stats
        }
    
    def search_clauses(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for clauses related to the query.
        
        Returns:
            List of dictionaries containing chunk information and scores
        """
        if not self.is_ready:
            raise ValueError(" Must process markdown content first using process_markdown()")
        
        results = self.searcher.find_similar_chunks(query, top_k)
        
        formatted_results = []
        for chunk, combined_score, score_details in results:
            formatted_results.append({
                'section_num': chunk.get('section_num', 'unknown'),
                'title': chunk.get('title', 'Untitled'),
                'text': chunk['text'],
                'word_count': chunk.get('word_count', 0),
                'type': chunk.get('type', 'unknown'),
                'num_subsections': chunk.get('num_subsections', 0),
                'scores': score_details
            })
        
        return formatted_results


if __name__ == "__main__":
    # Test the clause extractor with sample text
    sample_markdown = """
# Sample Contract

## 1. DEFINITIONS
This section defines key terms used throughout the agreement.

## 2. SCOPE OF WORK
The contractor shall perform the following services:
- Service A
- Service B

## 3. PAYMENT TERMS
### 3.1 Payment Schedule
Payment shall be made within 30 days of invoice.

### 3.2 Late Payments
Late payments will incur a 2% monthly fee.

## 4. TERMINATION
Either party may terminate this agreement with 30 days notice.
"""
    
    print("Testing Contract Clause Extractor...")
    extractor = ContractClauseExtractor()
    
    # Process sample markdown
    result = extractor.process_markdown(sample_markdown)
    print(f"Processing result: {result}")
    
    # Search for clauses
    if result['status'] == 'success':
        results = extractor.search_clauses("payment terms", top_k=3)
        print(f"\nSearch results for 'payment terms':")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (Score: {result['scores']['combined']:.3f})")
