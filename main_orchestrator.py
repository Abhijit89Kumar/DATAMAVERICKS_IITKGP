#!/usr/bin/env python3
"""
Main Contract Processing Orchestrator
=====================================

This script orchestrates the complete contract processing pipeline:
1. Parses all PDFs in the Contracts directory using pdf_parser.py
2. Extracts and searches for key contract clauses using clause_extractor.py
3. Saves results (top 5 matches) to JSON files

Features:
- Batch processing of all contract PDFs
- Intelligent clause extraction and analysis
- Standardized contract queries (payment, termination, confidentiality, etc.)
- JSON output with detailed search results and scores
- Progress tracking and error handling

Author: AI Assistant
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
from datetime import datetime
import warnings
from dataclasses import asdict, is_dataclass
import numpy as np
import re
warnings.filterwarnings("ignore")

from huggingface_hub import login
import os

# Hugging Face login - use environment variable for token
# Set HUGGINGFACE_TOKEN environment variable with your token
# or comment out the login if not needed
if os.getenv('HF_TOKEN'):
    login(token=os.getenv('HF_TOKEN'))
else:
    print("Warning: HF_TOKEN environment variable not set. Some features may not work.")


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle numpy types and dataclasses.
    Fixes serialization issues with ClauseComparisonResult and float32 values.
    """
    
    def default(self, obj):
        # Handle dataclass objects (like ClauseComparisonResult)
        if is_dataclass(obj):
            return asdict(obj)
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Let the base class handle other types
        return super().default(obj)


# Import our custom modules
try:
    from pdf_parser import RobustPDFParser
    from clause_extractor import ContractClauseExtractor
    from contract_classification import ContractClauseClassifier
    from llm_query_generator import ContractQueryGenerator
except ImportError as e:
    print(f" Error importing modules: {e}")
    print("Make sure all required modules are in the same directory")
    sys.exit(1)


class ContractOrchestrator:
    """
    Main orchestrator class that manages the complete contract processing pipeline.
    """
    
    def __init__(self, enable_ocr: bool = True, ocr_language: str = 'en',
                 semantic_weight: float = 1, tfidf_weight: float = 0,
                 classification_threshold: float = 0.8):
        """
        Initialize the orchestrator with all processing components.
        """
        print("Initializing Enhanced Contract Processing Orchestrator")
        print("=" * 70)
        
        # Initialize core components
        self.pdf_parser = RobustPDFParser(enable_ocr=enable_ocr, ocr_language=ocr_language)
        self.clause_extractor = ContractClauseExtractor(
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight
        )
        
        # Initialize new components
        self.classifier = ContractClauseClassifier(
            classification_threshold=classification_threshold
        )
        self.query_generator = ContractQueryGenerator()
        
        # Storage for processed data
        self.contract_data = {}  # {contract_name: {clauses, metadata}}
        self.template_data = {}  # {template_name: {clauses, metadata}}
        
        # Standard contract queries - Specific to healthcare/insurance contracts
        self.standard_queries = [
            "Medicaid Timely Filing: This attribute ensures Medicaid claims are submitted and processed within 120 days of service, or from notification of retroactive eligibility, in compliance with regulatory and contractual requirements. Claims must use appropriate service codes and be submitted electronically (EDI) or via NUCC paper forms if needed. Providers must supply all necessary information for payment determination, and respond to additional requests within 60 days or the filing period, whichever is longer. Properly submitted clean claims are adjudicated and paid promptly according to Medicaid program rules and the members health plan.",
            "Medicare Timely Filing: This attribute ensures Medicare Advantage claims are submitted and processed within 90 days of service, or from notification of retroactive primary payor responsibility, in compliance with regulatory and Plan requirements. Claims must use appropriate service codes and be submitted electronically (EDI) or via NUCC paper forms if needed. Providers must provide all information necessary for payment determination and respond to additional requests within 60 days or the filing period, whichever is longer. Properly submitted clean claims are adjudicated and paid according to the member's health plan and applicable regulations.",
            "No Steerage/SOC: Elevance may create, modify, or discontinue networks, products, and programs based on quality or business needs. Providers participate only in designated networks after meeting credentialing and participation requirements. Certain services may be limited to specific providers in Health Services Designated Networks, and providers must refer members accordingly; reimbursement is not provided for unauthorized services.",
            "Medicaid Fee Schedule: The Rate for Covered Services will be calculated as a percentage of the Professional Provider Market Master Fee Schedule in effect on the service date. This schedule may be modified prospectively during the Agreement term. In line with the CRA, reimbursement will not be based on automatic escalators such as Medicare updates or inflation indexes unless approved by Medicaid Affiliate Services. Although Provider is not part of the Medicaid Affiliate Network, if services are rendered to a Medicaid Affiliate's Member, the Provider must accept the applicable state program's Medicaid rate as full payment, subject to regulatory requirements and prior authorization, except for emergencies or self-referred services. Coordination of necessary information between Provider and Medicaid Affiliate will be facilitated upon request.",
            "Medicare Fee Schedule: For Members in a Medicare Advantage Network, the Provider will accept as full payment the lesser of Eligible Charges or the Medicare Advantage Rate, minus applicable Cost Shares. The rate is based on the Medicare Fee Schedule and reflects per-claim payments under original Medicare, excluding bonuses or settlements. Payments automatically adjust for CMS directives or regulatory changes. Providers are not compensated for member bad debts, and payments are final based on the system logic and data at claim processing, with no retroactive adjustments."
        ]
        
        print("Orchestrator initialized successfully!")
        print(f" Standard queries loaded: {len(self.standard_queries)}")
        print("=" * 60)
    
    def clean_clause_text(self, text: str) -> str:
        """
        Simple and lightweight text cleaning for contract/template clauses.
        Applied only to top clauses before classification to improve similarity stability.
        
        Args:
            text: Raw clause text to clean
            
        Returns:
            Cleaned clause text
        """
        if not text or not text.strip():
            return text
        
        # 1. Clean markdown tables first
        cleaned = self._clean_markdown_tables(text)
        
        # 2. Normalize whitespace - replace multiple spaces/tabs with single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 3. Remove extra line breaks and normalize them
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
        
        # 4. Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in cleaned.split('\n')]
        cleaned = '\n'.join(lines)
        
        # 5. Remove random characters that might interfere (like extra punctuation clusters)
        cleaned = re.sub(r'[•·◦▪▫–—―]{2,}', ' ', cleaned)  # Remove multiple bullet points
        cleaned = re.sub(r'[\.]{3,}', '...', cleaned)      # Normalize multiple dots
        cleaned = re.sub(r'[-_]{3,}', '---', cleaned)      # Normalize multiple dashes
        
        # 6. Remove extra spaces around punctuation
        cleaned = re.sub(r'\s+([,.;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'([,.;:!?])\s+', r'\1 ', cleaned)
        
        # 7. Remove common OCR artifacts and random characters
        cleaned = re.sub(r'[^\w\s\.,;:!?()\[\]{}"\'`+*/=<>@#$%&-]', '', cleaned)
        
        # 8. Final whitespace cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _clean_markdown_tables(self, text: str) -> str:
        """
        Remove proper markdown tables completely from text (including all content).
        Only removes actual table structures, not text that happens to contain | characters.
        
        Args:
            text: Text that may contain markdown tables
            
        Returns:
            Text with proper markdown tables completely removed
        """
        if not text or '|' not in text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Check if this is a table separator line (like |---|---|)
            if re.match(r'^\|[\s\-\:\|]+\|$', line):
                in_table = True
                continue  # Skip separator line
            
            # Check if this looks like a proper markdown table row
            # Must start and end with | and have at least 2 | separators
            is_table_row = (line.startswith('|') and line.endswith('|') and 
                           line.count('|') >= 3)
            
            if is_table_row:
                # Check if it's a header row by looking at the next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r'^\|[\s\-\:\|]+\|$', next_line):
                        in_table = True
                        continue  # Skip table header
                
                # If we're already in a table, skip this row
                if in_table:
                    continue
            else:
                # Not a table row, reset table state
                in_table = False
                # Keep non-table lines as-is
                if original_line.strip():
                    cleaned_lines.append(original_line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def find_all_contracts(self) -> List[Path]:
        """
        Find all PDF files in the Contracts directory structure.
        """
        contracts_dir = Path("Contracts")
        # contracts_dir = Path("Standard Templates")
        pdf_files = []
        
        if not contracts_dir.exists():
            print(f" Error: {contracts_dir} directory not found!")
            return pdf_files
        
        # found_pdfs = list(contracts_dir.glob("*.pdf"))
        # pdf_files.extend(found_pdfs)
        # print(f"Found {len(found_pdfs)} PDF files in {contracts_dir}")
        
        # Look for PDF files in all subdirectories
        for subdir in ["TN", "WA"]:
            subdir_path = contracts_dir / subdir
            if subdir_path.exists():
                found_pdfs = list(subdir_path.glob("*.pdf"))
                pdf_files.extend(found_pdfs)
                print(f"Found {len(found_pdfs)} PDF files in {subdir_path}")
            else:
                print(f" Warning: {subdir_path} directory not found!")

        
        print(f" Total contracts found: {len(pdf_files)}")
        return pdf_files
    
    def find_all_templates(self) -> List[Path]:
        """
        Find all PDF files in the Standard Templates directory.
        """
        templates_dir = Path("Standard Templates")
        pdf_files = []
        
        if not templates_dir.exists():
            print(f" Error: {templates_dir} directory not found!")
            return pdf_files
        
        # Look for PDF files directly in the templates directory
        found_pdfs = list(templates_dir.glob("*.pdf"))
        pdf_files.extend(found_pdfs)
        print(f"Found {len(found_pdfs)} template PDF files in {templates_dir}")
        
        print(f" Total templates found: {len(pdf_files)}")
        return pdf_files
    
    def process_single_contract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single contract PDF through the complete pipeline.
        
        Returns:
            Dictionary with parsing results, search results, and metadata
        """
        contract_name = pdf_path.stem
        print(f"\n{'='*80}")
        print(f" PROCESSING CONTRACT: {contract_name}")
        print(f"Path: {pdf_path}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Step 1: Try to use existing parsed file first, then parse PDF
            markdown_content = None
            
            # Check for existing parsed markdown file
            existing_parsed_file = Path(f"parsed_outputs/{pdf_path.parent.name}/{contract_name}_parsed.md")
            if existing_parsed_file.exists():
                try:
                    print(" Found existing parsed file, using it...")
                    with open(existing_parsed_file, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    if len(markdown_content.strip()) >= 100:
                        print(f"Using existing parsed file ({len(markdown_content)} characters)")
                    else:
                        markdown_content = None
                        print(" Existing parsed file too short, will re-parse")
                except Exception as e:
                    print(f" Could not read existing parsed file: {e}")
                    markdown_content = None
            
            # Parse PDF if no existing file or existing file was too short
            if not markdown_content:
                print("\n STEP 1: PDF Parsing...")
                try:
                    markdown_content = self.pdf_parser.parse_pdf(str(pdf_path))
                except Exception as e:
                    print(f" PDF parsing failed with error: {e}")
                    if "MemoryError" in str(e) or "memory" in str(e).lower():
                        print(" Memory error detected - trying basic fallback extraction...")
                        markdown_content = self._fallback_pdf_extraction(pdf_path)
                    else:
                        markdown_content = None
            
            if not markdown_content or len(markdown_content.strip()) < 100:
                return {
                    "status": "error",
                    "contract": contract_name,
                    "error": "PDF parsing failed or content too short",
                    "processing_time": time.time() - start_time
                }
            
            print(f"PDF parsed successfully ({len(markdown_content)} characters)")
            
            # Step 2: Process markdown and prepare for search
            print("\n STEP 2: Markdown Processing and Chunking...")
            processing_result = self.clause_extractor.process_markdown(markdown_content)
            
            if processing_result["status"] != "success":
                return {
                    "status": "error",
                    "contract": contract_name,
                    "error": f"Markdown processing failed: {processing_result.get('message', 'Unknown error')}",
                    "processing_time": time.time() - start_time
                }
            
            print(f"Markdown processed successfully ({processing_result['chunks_created']} chunks)")
            
            # Step 3: Search for standard contract clauses
            print("\n STEP 3: Clause Search and Analysis...")
            search_results = {}
            contract_clauses = []
            
            for i, query in enumerate(self.standard_queries, 1):
                print(f"   Query {i}/{len(self.standard_queries)}: {query[:50]}...")
                
                try:
                    results = self.clause_extractor.search_clauses(query, top_k=5)
                    search_results[query] = results
                    
                    if results:
                        # Get the top 1 clause for each query
                        top_clause = results[0]
                        
                        # Determine contract type from path or filename
                        contract_type = "TN" if "/TN/" in str(pdf_path) or "TN" in contract_name.upper() else "WA"
                        
                        # Clean the top clause text before storing for classification
                        cleaned_clause_text = self.clean_clause_text(top_clause['text'])
                        if len(cleaned_clause_text) != len(top_clause['text']):
                            print(f"     Text cleaned: {len(top_clause['text'])} -> {len(cleaned_clause_text)} characters")
                        
                        contract_clauses.append({
                            "text": cleaned_clause_text,
                            "type": contract_type,
                            "query": query,
                            "query_index": i,
                            "score": top_clause['scores']['combined'],
                            "section_num": top_clause.get('section_num', 'unknown'),
                            "title": top_clause.get('title', 'Untitled')
                        })
                        
                        best_score = results[0]['scores']['combined']
                        print(f"     Found {len(results)} matches (best: {best_score:.3f})")
                    else:
                        print(f"      No matches found")
                        
                except Exception as e:
                    print(f"      Search failed: {e}")
                    search_results[query] = []
            
            processing_time = time.time() - start_time
            
            # Store contract data for later use
            self.contract_data[contract_name] = {
                "clauses": contract_clauses,
                "metadata": {
                    "type": "TN" if "/TN/" in str(pdf_path) or "TN" in contract_name.upper() else "WA",
                    "processing_time": processing_time,
                    "total_clauses": len(contract_clauses)
                }
            }
            
            # Compile final results
            result = {
                "status": "success",
                "contract": contract_name,
                "file_path": str(pdf_path),
                "processing_time": processing_time,
                "contract_type": "TN" if "/TN/" in str(pdf_path) or "TN" in contract_name.upper() else "WA",
                "statistics": processing_result["statistics"],
                "search_results": search_results,
                "contract_clauses": contract_clauses,
                "queries_processed": len(self.standard_queries),
                "total_matches_found": sum(len(results) for results in search_results.values()),
                "processed_at": datetime.now().isoformat()
            }
            
            print(f"\nCONTRACT PROCESSING COMPLETED")
            print(f"    Processing time: {processing_time:.2f} seconds")
            print(f"   Total matches found: {result['total_matches_found']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing {contract_name}: {e}"
            print(f" {error_msg}")
            traceback.print_exc()
            
            return {
                "status": "error",
                "contract": contract_name,
                "file_path": str(pdf_path),
                "error": error_msg,
                "processing_time": time.time() - start_time,
                "processed_at": datetime.now().isoformat()
            }
    
    def process_single_template(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single template PDF through the complete pipeline.
        IDENTICAL processing logic as contracts - just different path handling.
        
        Returns:
            Dictionary with parsing results, search results, and metadata
        """
        template_name = pdf_path.stem
        print(f"\n{'='*80}")
        print(f" PROCESSING TEMPLATE: {template_name}")
        print(f"Path: {pdf_path}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Step 1: Try to use existing parsed file first, then parse PDF (SAME AS CONTRACTS)
            markdown_content = None
            
            # Check for existing parsed markdown file - for templates, look in parsed_outputs/Templates/
            existing_parsed_file = Path(f"parsed_outputs/Templates/{template_name}_parsed.md")
            if existing_parsed_file.exists():
                try:
                    print(" Found existing parsed template file, using it...")
                    with open(existing_parsed_file, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    if len(markdown_content.strip()) >= 100:
                        print(f"Using existing parsed template file ({len(markdown_content)} characters)")
                    else:
                        markdown_content = None
                        print(" Existing parsed template file too short, will re-parse")
                except Exception as e:
                    print(f" Could not read existing parsed template file: {e}")
                    markdown_content = None
            
            # Parse PDF if no existing file or existing file was too short (SAME AS CONTRACTS)
            if not markdown_content:
                print("\n STEP 1: Template PDF Parsing...")
                try:
                    markdown_content = self.pdf_parser.parse_pdf(str(pdf_path))
                except Exception as e:
                    print(f" Template PDF parsing failed with error: {e}")
                    if "MemoryError" in str(e) or "memory" in str(e).lower():
                        print(" Memory error detected - trying basic fallback extraction...")
                        markdown_content = self._fallback_pdf_extraction(pdf_path)
                    else:
                        markdown_content = None
            
            if not markdown_content or len(markdown_content.strip()) < 100:
                return {
                    "status": "error",
                    "template": template_name,
                    "error": "Template PDF parsing failed or content too short",
                    "processing_time": time.time() - start_time
                }
            
            print(f"Template PDF parsed successfully ({len(markdown_content)} characters)")
            
            # Step 2: Process markdown and prepare for search (SAME AS CONTRACTS)
            print("\n STEP 2: Template Markdown Processing and Chunking...")
            processing_result = self.clause_extractor.process_markdown(markdown_content)
            
            if processing_result["status"] != "success":
                return {
                    "status": "error",
                    "template": template_name,
                    "error": f"Template markdown processing failed: {processing_result.get('message', 'Unknown error')}",
                    "processing_time": time.time() - start_time
                }
            
            print(f"Template markdown processed successfully ({processing_result['chunks_created']} chunks)")
            
            # Step 3: Search for standard contract clauses (SAME AS CONTRACTS)
            print("\n STEP 3: Template Clause Search and Analysis...")
            search_results = {}
            template_clauses = []
            
            for i, query in enumerate(self.standard_queries, 1):
                print(f"   Query {i}/{len(self.standard_queries)}: {query[:50]}...")
                
                try:
                    results = self.clause_extractor.search_clauses(query, top_k=5)
                    search_results[query] = results
                    
                    if results:
                        # Get the top 1 clause for each query
                        top_clause = results[0]
                        
                        # Determine template type from filename
                        template_type = "TN" if "TN" in template_name.upper() else "WA"
                        
                        # Clean the top template clause text before storing for classification
                        cleaned_template_text = self.clean_clause_text(top_clause['text'])
                        if len(cleaned_template_text) != len(top_clause['text']):
                            print(f"     Template text cleaned: {len(top_clause['text'])} -> {len(cleaned_template_text)} characters")
                        
                        template_clauses.append({
                            "text": cleaned_template_text,
                            "type": template_type,
                            "query": query,
                            "query_index": i,
                            "score": top_clause['scores']['combined'],
                            "section_num": top_clause.get('section_num', 'unknown'),
                            "title": top_clause.get('title', 'Untitled')
                        })
                        
                        best_score = results[0]['scores']['combined']
                        print(f"     Found {len(results)} matches (best: {best_score:.3f})")
                    else:
                        print(f"      No matches found")
                        
                except Exception as e:
                    print(f"      Search failed: {e}")
                    search_results[query] = []
            
            processing_time = time.time() - start_time
            
            # Store template data for later use (SAME STRUCTURE AS CONTRACTS)
            self.template_data[template_name] = {
                "clauses": template_clauses,
                "metadata": {
                    "type": "TN" if "TN" in template_name.upper() else "WA",
                    "processing_time": processing_time,
                    "total_clauses": len(template_clauses)
                }
            }
            
            # Compile final results (SAME STRUCTURE AS CONTRACTS)
            result = {
                "status": "success",
                "template": template_name,
                "file_path": str(pdf_path),
                "processing_time": processing_time,
                "template_type": "TN" if "TN" in template_name.upper() else "WA",
                "statistics": processing_result["statistics"],
                "search_results": search_results,
                "template_clauses": template_clauses,
                "queries_processed": len(self.standard_queries),
                "total_matches_found": sum(len(results) for results in search_results.values()),
                "processed_at": datetime.now().isoformat()
            }
            
            print(f"\nTEMPLATE PROCESSING COMPLETED")
            print(f"    Processing time: {processing_time:.2f} seconds")
            print(f"   Total matches found: {result['total_matches_found']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing template {template_name}: {e}"
            print(f" {error_msg}")
            traceback.print_exc()
            
            return {
                "status": "error",
                "template": template_name,
                "file_path": str(pdf_path),
                "error": error_msg,
                "processing_time": time.time() - start_time,
                "processed_at": datetime.now().isoformat()
            }
    
    def _fallback_pdf_extraction(self, pdf_path: Path) -> Optional[str]:
        """
        Fallback PDF extraction using basic PyMuPDF when Docling fails.
        """
        try:
            import fitz  # PyMuPDF
            print(" Using basic PyMuPDF extraction as fallback...")
            
            doc = fitz.open(str(pdf_path))
            markdown_content = f"# Document: {pdf_path.name}\n\n"
            markdown_content += "---\n\n"
            markdown_content += "## Document Content (via PyMuPDF Fallback)\n\n"
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        markdown_content += f"### Page {page_num + 1}\n\n{page_text}\n\n"
                except Exception as e:
                    print(f" Failed to extract text from page {page_num + 1}: {e}")
            
            doc.close()
            print(f"Fallback extraction successful ({len(markdown_content)} characters)")
            return markdown_content
            
        except Exception as e:
            print(f" Fallback extraction also failed: {e}")
            return None
    
    def demonstrate_llm_query_generation(self) -> Dict[str, Any]:
        """
        Demonstrate LLM query generation capabilities.
        
        Returns:
            Dictionary containing demonstration results
        """
        print("\n DEMONSTRATING LLM QUERY GENERATION")
        print("=" * 60)
        
        try:
            demo_result = self.query_generator.demonstrate_query_generation()
            
            if demo_result["status"] == "success":
                print("LLM query generation demonstration successful")
                print(f" Model used: {demo_result.get('llm_model', 'Unknown')}")
                
                # Show a few sample generated queries
                if "generated_queries" in demo_result:
                    tn_queries = demo_result["generated_queries"].get("TN_queries", [])
                    wa_queries = demo_result["generated_queries"].get("WA_queries", [])
                    
                    if tn_queries:
                        print(f" Sample TN queries generated: {len(tn_queries)}")
                        for i, query in enumerate(tn_queries[:2], 1):
                            print(f"   {i}. {query[:80]}...")
                    
                    if wa_queries:
                        print(f" Sample WA queries generated: {len(wa_queries)}")
                        for i, query in enumerate(wa_queries[:2], 1):
                            print(f"   {i}. {query[:80]}...")
            else:
                print(f" LLM query generation not available: {demo_result.get('message', 'Unknown error')}")
            
            return demo_result
            
        except Exception as e:
            error_msg = f"LLM demonstration failed: {e}"
            print(f" {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
    
    def perform_classification_analysis(self) -> Dict[str, Any]:
        """
        Perform classification analysis comparing contract clauses with template clauses.
        
        Returns:
            Dictionary containing classification results
        """
        print("\nPERFORMING CLASSIFICATION ANALYSIS")
        print("=" * 60)
        
        if not self.contract_data or not self.template_data:
            return {
                "status": "error",
                "message": "Contract and template data must be processed first"
            }
        
        try:
            all_contract_clauses = []
            all_template_clauses = []
            
            # Collect all contract clauses
            for contract_name, data in self.contract_data.items():
                all_contract_clauses.extend(data["clauses"])
            
            # Collect all template clauses
            for template_name, data in self.template_data.items():
                all_template_clauses.extend(data["clauses"])
            
            print(f" Comparing {len(all_contract_clauses)} contract clauses with {len(all_template_clauses)} template clauses")
            
            # Perform batch classification
            classification_results = self.classifier.batch_compare_clauses(
                contract_clauses=all_contract_clauses,
                template_clauses=all_template_clauses
            )
            
            # Generate classification summary
            classification_summary = self.classifier.get_classification_summary(classification_results)
            
            print(f"Classification analysis complete")
            print(f"   Standard clauses: {classification_summary.get('standard_clauses', 0)}")
            print(f"   Non-standard clauses: {classification_summary.get('non_standard_clauses', 0)}")
            print(f"   Average similarity: {classification_summary.get('average_similarity', 0):.3f}")
            
            return {
                "status": "success",
                "classification_results": classification_results,
                "classification_summary": classification_summary
            }
            
        except Exception as e:
            error_msg = f"Classification analysis failed: {e}"
            print(f" {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
    
    def generate_llm_explanations(self, classification_results: List) -> Dict[str, Any]:
        """
        Generate LLM-based explanations for non-standard clause classifications.
        
        Args:
            classification_results: List of classification result objects
            
        Returns:
            Dictionary containing LLM explanation results
        """
        print("\n GENERATING LLM EXPLANATIONS FOR NON-STANDARD CLAUSES")
        print("=" * 60)
        
        try:
            # Quick LLM availability check at the start
            if not hasattr(self.query_generator, 'llm') or not self.query_generator.llm:
                print("LLM not available - skipping explanations")
                return {
                    "status": "skipped",
                    "explanations": [],
                    "explanations_count": 0,
                    "message": "LLM not available"
                }
            
            # Filter for non-standard classifications only
            non_standard_results = [
                result for result in classification_results 
                if hasattr(result, 'classification') and result.classification == 'non_standard'
            ]
            
            if not non_standard_results:
                print("No non-standard clauses found - skipping LLM explanations")
                return {
                    "status": "success",
                    "explanations": [],
                    "explanations_count": 0,
                    "message": "No non-standard clauses to explain"
                }
            
            print(f" Found {len(non_standard_results)} non-standard clauses to explain")
            print(f"LLM is available and ready for explanations")
            
            explanations = []
            
            for i, result in enumerate(non_standard_results, 1):
                print(f"Generating explanation {i}/{len(non_standard_results)}...")
                
                try:
                    explanation = self._generate_single_llm_explanation(result)
                    if explanation:
                        explanations.append(explanation)
                        print(f"Explanation generated successfully")
                    else:
                        print(f"Failed to generate explanation")
                        
                except Exception as e:
                    print(f"Error generating single LLM explanation: {e}")
                    continue
            
            print(f"LLM explanations complete")
            print(f"Explanations generated: {len(explanations)}")
            
            return {
                "status": "success",
                "explanations": explanations,
                "explanations_count": len(explanations),
                "total_non_standard": len(non_standard_results)
            }
            
        except Exception as e:
            error_msg = f"LLM explanation generation failed: {e}"
            print(f" {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "explanations": [],
                "explanations_count": 0
            }
    
    def _generate_single_llm_explanation(self, classification_result) -> Optional[Dict[str, Any]]:
        """
        Generate a single LLM explanation for a non-standard clause.
        
        Args:
            classification_result: Single classification result object
            
        Returns:
            Dictionary containing the explanation
        """
        try:
            # Extract data from classification result
            contract_clause = classification_result.contract_clause
            template_clause = classification_result.template_clause
            contract_type = classification_result.contract_type
            query_name = classification_result.query_name
            
            # Get similarity scores
            cosine_sim = getattr(classification_result, 'cosine', 0.0)
            semantic_sim = getattr(classification_result, 'semantic_similarity', 0.0)
            jaccard_sim = getattr(classification_result, 'jaccard', 0.0)
            avg_similarity = getattr(classification_result, 'average_similarity', 0.0)
            confidence = getattr(classification_result, 'confidence_score', 0.0)
            
            # Create structured prompt for LLM
            prompt = self._create_explanation_prompt(
                query_name=query_name,
                contract_clause=contract_clause,
                template_clause=template_clause,
                contract_type=contract_type,
                cosine_sim=cosine_sim,
                semantic_sim=semantic_sim,
                jaccard_sim=jaccard_sim,
                avg_similarity=avg_similarity,
                confidence=confidence
            )
            
            # Generate explanation using LLM
            llm_response = self._call_llm_for_explanation(prompt)
            
            if llm_response:
                return {
                    "contract_type": contract_type,
                    "query_name": query_name,
                    "contract_clause": contract_clause[:200] + "..." if len(contract_clause) > 200 else contract_clause,
                    "template_clause": template_clause[:200] + "..." if len(template_clause) > 200 else template_clause,
                    "similarity_scores": {
                        "cosine": float(cosine_sim),
                        "semantic": float(semantic_sim) if semantic_sim else None,
                        "jaccard": float(jaccard_sim),
                        "average": float(avg_similarity),
                        "confidence": float(confidence)
                    },
                    "llm_explanation": llm_response,
                    "classification": "non_standard"
                }
            
            return None
            
        except Exception as e:
            print(f"Error generating single LLM explanation: {e}")
            return None
    
    def _create_explanation_prompt(self, query_name: str, contract_clause: str, 
                                 template_clause: str, contract_type: str,
                                 cosine_sim: float, semantic_sim: float, jaccard_sim: float,
                                 avg_similarity: float, confidence: float) -> str:
        """
        Create a structured prompt for LLM explanation generation.
        """
        prompt = f"""You are an expert contract analyst specializing in healthcare contract compliance. Your task is to explain why a contract clause was classified as NON-STANDARD compared to the template.

**ANALYSIS CONTEXT:**
- Contract Type: {contract_type}
- Query Category: {query_name}
- Classification: NON-STANDARD
- Average Similarity Score: {avg_similarity:.3f}
- Confidence Score: {confidence:.3f}

**SIMILARITY SCORES:**
- Cosine Similarity: {cosine_sim:.3f}
- Semantic Similarity: {f"{semantic_sim:.3f}" if semantic_sim is not None else 'N/A'}
- Jaccard Similarity: {jaccard_sim:.3f}

**TEMPLATE CLAUSE (STANDARD):**
{template_clause}

**CONTRACT CLAUSE (NON-STANDARD):**
{contract_clause}

**INSTRUCTIONS:**
Please provide a detailed explanation of why this contract clause was classified as NON-STANDARD. Your explanation should:

1. **Key Differences**: Identify the specific differences between the contract clause and the template clause
2. **Missing Elements**: Point out any important standard elements that are missing from the contract clause
3. **Additional Elements**: Highlight any extra or unusual terms in the contract clause not found in the template
4. **Compliance Impact**: Explain the potential impact of these differences on contract compliance or risk
5. **Severity Assessment**: Rate the severity of the non-standardness (Low/Medium/High) and justify your rating

**FORMAT YOUR RESPONSE AS:**

**Key Differences:**
[List the main differences]

**Missing Elements:**
[List what's missing from standard template]

**Additional Elements:**
[List extra terms in contract clause]

**Severity Assessment:**
[Low/Medium/High] - [Justification]

**Summary:**
[Concise summary of why this clause is non-standard]

Please be specific, concise, and focus on actionable insights for contract management. Never Use Emojis!!"""

        return prompt
    
    def _call_llm_for_explanation(self, prompt: str) -> Optional[str]:
        """
        Call LLM to generate explanation with low temperature for consistency.
        
        Args:
            prompt: Structured prompt for explanation
            
        Returns:
            LLM response or None if failed
        """
        try:
            # Check if query generator is available and has LLM capability
            if not hasattr(self.query_generator, 'llm') or not self.query_generator.llm:
                print("LLM not available for explanations")
                return None

            print(f"Calling LLM for explanation...")

            # Check LLM status before calling
            if not callable(self.query_generator.llm):
                print("LLM is not callable")
                return None
            
            # Generate explanation with reduced max_tokens to prevent hanging
            response = self.query_generator.llm(
                prompt,
                temperature=0.1,  # Very low for consistency
                max_tokens=1500,   # Reduced from 3000 to prevent hanging
            )

            print(f"LLM response received")

            # Handle llama-cpp-python response format
            if response and isinstance(response, dict):
                if 'choices' in response and len(response['choices']) > 0:
                    text = response['choices'][0]['text'].strip()
                    print(f"Explanation length: {len(text)} characters")
                    return text
            elif isinstance(response, str):
                text = response.strip()
                print(f"Explanation length: {len(text)} characters")
                return text
            
            print(f"No valid response from LLM")
            return None
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    

    def create_output_directory(self) -> Path:
        """Create output directory for results."""
        output_dir = Path("contract_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / f"analysis_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        
        return session_dir
    
    def save_results(self, 
                    contract_results: List[Dict[str, Any]], 
                    template_results: List[Dict[str, Any]],
                    llm_demo_result: Dict[str, Any],
                    classification_result: Dict[str, Any],
                    llm_explanation_result: Dict[str, Any],
                    output_dir: Path) -> None:
        """Save all processing results to JSON files."""
        print(f"\n SAVING ENHANCED RESULTS TO: {output_dir}")
        print("=" * 70)
        
        # Save individual contract results
        print(" Saving contract analysis results...")
        for result in contract_results:
            contract_name = result["contract"]
            filename = f"{contract_name}_analysis.json"
            filepath = output_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                print(f"Saved contract results: {filename}")
            except Exception as e:
                print(f" Failed to save {filename}: {e}")
        
        # Save individual template results
        print(" Saving template analysis results...")
        for result in template_results:
            template_name = result["template"]
            filename = f"{template_name}_template_analysis.json"
            filepath = output_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                print(f"Saved template results: {filename}")
            except Exception as e:
                print(f" Failed to save {filename}: {e}")
        
        # Save LLM demonstration results
        if llm_demo_result:
            llm_file = output_dir / "llm_query_generation_demo.json"
            try:
                with open(llm_file, 'w', encoding='utf-8') as f:
                    json.dump(llm_demo_result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                print(f"Saved LLM demo results: llm_query_generation_demo.json")
            except Exception as e:
                print(f" Failed to save LLM demo results: {e}")
        
        # Save classification results
        if classification_result.get("status") == "success":
            classification_file = output_dir / "classification_analysis.json"
            try:
                with open(classification_file, 'w', encoding='utf-8') as f:
                    json.dump(classification_result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                print(f"Saved classification results: classification_analysis.json")
            except Exception as e:
                print(f" Failed to save classification results: {e}")
        
        # Save LLM explanation results
        if llm_explanation_result.get("status") == "success":
            # Create LLM explanations subdirectory
            llm_explanations_dir = output_dir / "llm_explanations"
            llm_explanations_dir.mkdir(exist_ok=True)
            
            try:
                # Save detailed explanations
                explanations_file = llm_explanations_dir / "non_standard_explanations.json"
                with open(explanations_file, 'w', encoding='utf-8') as f:
                    json.dump(llm_explanation_result, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
                
                # Create human-readable summary
                if llm_explanation_result.get("explanations"):
                    summary_file = llm_explanations_dir / "explanations_summary.txt"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write("LLM Explanations for Non-Standard Clauses\n")
                        f.write("=" * 50 + "\n\n")
                        
                        for i, explanation in enumerate(llm_explanation_result["explanations"], 1):
                            f.write(f"{i}. {explanation['contract_type']} - {explanation['query_name']}\n")
                            f.write(f"Average Similarity: {explanation['similarity_scores']['average']:.3f}\n")
                            f.write("-" * 40 + "\n")
                            f.write(f"{explanation['llm_explanation']}\n")
                            f.write("=" * 50 + "\n\n")
                
                print(f"Saved LLM explanations: llm_explanations/")
            except Exception as e:
                print(f" Failed to save LLM explanations: {e}")
        elif llm_explanation_result.get("status") == "success" and llm_explanation_result.get("explanations_count") == 0:
            print(f"No non-standard clauses found - no LLM explanations needed")
        
        
        # Save comprehensive summary
        summary = {
            "session_info": {
                "processed_at": datetime.now().isoformat(),
                "total_contracts": len(contract_results),
                "successful_contracts": len([r for r in contract_results if r["status"] == "success"]),
                "failed_contracts": len([r for r in contract_results if r["status"] == "error"]),
                "total_templates": len(template_results),
                "successful_templates": len([r for r in template_results if r["status"] == "success"]),
                "failed_templates": len([r for r in template_results if r["status"] == "error"]),
                "standard_queries": self.standard_queries,
                "components_status": {
                    "llm_query_generation": llm_demo_result.get("status", "not_run"),
                    "classification_analysis": classification_result.get("status", "not_run"),
                    "llm_explanations": llm_explanation_result.get("status", "not_run")
                }
            },
            "contract_results": contract_results,
            "template_results": template_results,
            "analysis_results": {
                "llm_demonstration": llm_demo_result,
                "classification": classification_result.get("classification_summary", {}),
                "llm_explanations": {
                    "status": llm_explanation_result.get("status", "not_run"),
                    "explanations_count": llm_explanation_result.get("explanations_count", 0),
                    "non_standard_clauses_found": llm_explanation_result.get("total_non_standard", 0)
                }
            }
        }
        
        summary_file = output_dir / "comprehensive_analysis_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            print(f"Saved comprehensive summary: comprehensive_analysis_summary.json")
        except Exception as e:
            print(f" Failed to save comprehensive summary: {e}")
        
        # Create enhanced CSV report
        self.create_enhanced_csv_report(contract_results, template_results, 
                                      classification_result, output_dir)
    
    def create_enhanced_csv_report(self, 
                                  contract_results: List[Dict[str, Any]], 
                                  template_results: List[Dict[str, Any]],
                                  classification_result: Dict[str, Any],
                                  output_dir: Path) -> None:
        """Create enhanced CSV reports with key metrics."""
        try:
            import pandas as pd
            
            # Contract Analysis Report
            print("Creating contract analysis CSV report...")
            contract_data = []
            for result in contract_results:
                if result["status"] == "success":
                    # Calculate top scores for each query
                    query_scores = {}
                    for query, search_results in result["search_results"].items():
                        if search_results:
                            top_score = search_results[0]["scores"]["combined"]
                            query_scores[f"Score: {query[:30]}..."] = f"{top_score:.3f}"
                        else:
                            query_scores[f"Score: {query[:30]}..."] = "0.000"
                    
                    row = {
                        "Contract": result["contract"],
                        "Type": result.get("contract_type", "Unknown"),
                        "Status": result["status"],
                        "Processing Time (s)": f"{result['processing_time']:.2f}",
                        "Total Chunks": result["statistics"]["total_chunks"],
                        "Clauses Extracted": len(result.get("contract_clauses", [])),
                        "Total Matches": result["total_matches_found"],
                        **query_scores
                    }
                else:
                    row = {
                        "Contract": result["contract"],
                        "Type": result.get("contract_type", "Unknown"),
                        "Status": result["status"],
                        "Error": result.get("error", "Unknown"),
                        "Processing Time (s)": f"{result['processing_time']:.2f}"
                    }
                
                contract_data.append(row)
            
            df_contracts = pd.DataFrame(contract_data)
            contracts_csv = output_dir / "contracts_analysis_report.csv"
            df_contracts.to_csv(contracts_csv, index=False)
            print(f"Saved contracts CSV report: contracts_analysis_report.csv")
            
            # Template Analysis Report
            print("Creating template analysis CSV report...")
            template_data = []
            for result in template_results:
                if result["status"] == "success":
                    row = {
                        "Template": result["template"],
                        "Type": result.get("template_type", "Unknown"),
                        "Status": result["status"],
                        "Processing Time (s)": f"{result['processing_time']:.2f}",
                        "Clauses Extracted": result.get("clauses_extracted", 0)
                    }
                else:
                    row = {
                        "Template": result["template"],
                        "Type": result.get("template_type", "Unknown"),
                        "Status": result["status"],
                        "Error": result.get("error", "Unknown"),
                        "Processing Time (s)": f"{result['processing_time']:.2f}"
                    }
                
                template_data.append(row)
            
            df_templates = pd.DataFrame(template_data)
            templates_csv = output_dir / "templates_analysis_report.csv"
            df_templates.to_csv(templates_csv, index=False)
            print(f"Saved templates CSV report: templates_analysis_report.csv")
            
            # Classification Summary Report
            if classification_result.get("status") == "success":
                print("Creating classification summary CSV report...")
                classification_summary = classification_result.get("classification_summary", {})
                
                summary_data = [{
                    "Metric": "Total Clauses Compared",
                    "Value": classification_summary.get("total_clauses", 0)
                }, {
                    "Metric": "Standard Clauses",
                    "Value": classification_summary.get("standard_clauses", 0)
                }, {
                    "Metric": "Non-Standard Clauses", 
                    "Value": classification_summary.get("non_standard_clauses", 0)
                }, {
                    "Metric": "Standard Percentage",
                    "Value": f"{classification_summary.get('standard_percentage', 0):.1f}%"
                }, {
                    "Metric": "Average Similarity Score",
                    "Value": f"{classification_summary.get('average_similarity', 0):.3f}"
                }]
                
                # Add contract type breakdown
                by_type = classification_summary.get("by_contract_type", {})
                for contract_type, type_data in by_type.items():
                    summary_data.extend([{
                        "Metric": f"{contract_type} - Total Clauses",
                        "Value": type_data.get("total", 0)
                    }, {
                        "Metric": f"{contract_type} - Standard Clauses",
                        "Value": type_data.get("standard", 0)
                    }, {
                        "Metric": f"{contract_type} - Average Similarity",
                        "Value": f"{type_data.get('avg_similarity', 0):.3f}"
                    }])
                
                df_classification = pd.DataFrame(summary_data)
                classification_csv = output_dir / "classification_summary_report.csv"
                df_classification.to_csv(classification_csv, index=False)
                print(f"Saved classification CSV report: classification_summary_report.csv")
            
        except ImportError:
            print(" pandas not available - skipping CSV reports")
        except Exception as e:
            print(f" Failed to create enhanced CSV reports: {e}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete enhanced contract analysis pipeline.
        """
        print("STARTING ENHANCED CONTRACT ANALYSIS PIPELINE")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Find all contracts and templates
        print("\nSTEP 1: Finding Files...")
        contracts = self.find_all_contracts()
        templates = self.find_all_templates()
        
        if not contracts:
            print(" No contracts found to process!")
            return {"status": "error", "message": "No contracts found"}
        
        if not templates:
            print(" No templates found to process!")
            return {"status": "error", "message": "No templates found"}
        
        # Step 2: Create output directory
        output_dir = self.create_output_directory()
        print(f"Output directory: {output_dir}")
        
        # Step 3: Demonstrate LLM Query Generation
        print(f"\n STEP 2: LLM Query Generation Demonstration...")
        llm_demo_result = self.demonstrate_llm_query_generation()
        
        # Step 4: Process templates first
        print(f"\n STEP 3: Processing Templates ({len(templates)} templates)")
        template_results = []
        template_success_count = 0
        template_error_count = 0
        
        for i, template_path in enumerate(templates, 1):
            print(f"\n PROCESSING TEMPLATE {i}/{len(templates)}")
            
            result = self.process_single_template(template_path)
            template_results.append(result)
            
            if result["status"] == "success":
                template_success_count += 1
            else:
                template_error_count += 1
        
        # Step 5: Process contracts
        print(f"\n STEP 4: Processing Contracts ({len(contracts)} contracts)")
        contract_results = []
        contract_success_count = 0
        contract_error_count = 0
        
        for i, contract_path in enumerate(contracts, 1):
            print(f"\n PROCESSING CONTRACT {i}/{len(contracts)}")
            
            result = self.process_single_contract(contract_path)
            contract_results.append(result)
            
            if result["status"] == "success":
                contract_success_count += 1
            else:
                contract_error_count += 1
        
        # Step 6: Perform Classification Analysis
        print(f"\nSTEP 5: Classification Analysis...")
        classification_result = self.perform_classification_analysis()
        
        # Step 7: Generate LLM Explanations for Non-Standard Clauses
        llm_explanation_result = {"status": "not_run", "message": "Skipped"}
        if classification_result.get("status") == "success":
            print(f"\n STEP 6: LLM Explanations for Non-Standard Clauses...")
            try:
                llm_explanation_result = self.generate_llm_explanations(
                    classification_result.get("classification_results", [])
                )
                print(f"LLM explanations completed: {llm_explanation_result.get('status', 'unknown')}")
            except Exception as e:
                print(f"LLM explanation step failed: {e}")
                llm_explanation_result = {"status": "error", "message": f"LLM step failed: {e}"}
        else:
            print(f"\n STEP 6: Skipping LLM explanations (classification failed)")
        
        # Step 7: Save all results
        print(f"\n STEP 7: Saving Results...")
        self.save_results(
            contract_results=contract_results,
            template_results=template_results,
            llm_demo_result=llm_demo_result,
            classification_result=classification_result,
            llm_explanation_result=llm_explanation_result,
            output_dir=output_dir
        )
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\nENHANCED ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f" Templates processed: {template_success_count}/{len(templates)} successful")
        print(f" Contracts processed: {contract_success_count}/{len(contracts)} successful")
        print(f" LLM demo: {llm_demo_result.get('status', 'unknown')}")
        print(f"Classification: {classification_result.get('status', 'unknown')}")
        print(f"LLM explanations: {llm_explanation_result.get('status', 'unknown')} ({llm_explanation_result.get('explanations_count', 0)} non-standard clauses explained)")
        print(f" Total processing time: {total_time:.2f} seconds")
        print(f" Results saved to: {output_dir}")
        print("=" * 80)
        
        return {
            "status": "complete",
            "processing_summary": {
                "total_contracts": len(contracts),
                "successful_contracts": contract_success_count,
                "failed_contracts": contract_error_count,
                "total_templates": len(templates),
                "successful_templates": template_success_count,
                "failed_templates": template_error_count,
                "total_time": total_time
            },
            "analysis_results": {
                "llm_demonstration": llm_demo_result,
                "classification": classification_result,
                "llm_explanations": llm_explanation_result
            },
            "output_directory": str(output_dir),
            "contract_results": contract_results,
            "template_results": template_results
        }


def main():
    """Main execution function."""
    try:
        # Initialize orchestrator
        orchestrator = ContractOrchestrator(
            enable_ocr=True,
            semantic_weight=0.85,
            tfidf_weight=0.15
        )
        
        # Run complete analysis
        final_results = orchestrator.run_complete_analysis()
        
        if final_results["status"] == "complete":
            print(f"\nSUCCESS! All contract analysis completed.")
            print(f"Check the results in: {final_results['output_directory']}")
        else:
            print(f"\n Analysis failed: {final_results.get('message', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n Analysis interrupted by user")
    except Exception as e:
        print(f"\n Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
