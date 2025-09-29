#!/usr/bin/env python3
"""
LLM Query Generator Module
==========================

This module provides LLM-based query generation for contract analysis.
Based on the inference.py structure, it loads and uses a local GGUF model
to generate contract analysis queries.

Features:
- Local GGUF model loading using llama-cpp-python
- Contract query generation
- Hybrid CPU-GPU inference support
- Backend-ready response structure

Author: AI Assistant
"""

import os
import sys
import glob
import logging
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama
    LLM_AVAILABLE = True
    print("LLM libraries available")
except ImportError:
    LLM_AVAILABLE = False
    print(" LLM libraries not available - query generation will be disabled")


class ContractQueryGenerator:
    """
    LLM-based query generator for contract analysis.
    """
    
    def __init__(self, 
                 model_repo: str = "unsloth/gemma-3n-E2B-it-GGUF",
                 local_dir: str = "./gemma_model",
                 model_pattern: str = "gemma-3n-E2B-it-Q4_0.gguf",
                 n_gpu_layers: int = 10,
                 n_threads: int = 8,
                 n_ctx: int = 2048):
        """
        Initialize the LLM query generator.
        
        Args:
            model_repo: Hugging Face model repository
            local_dir: Local directory to store the model
            model_pattern: Pattern to match GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of CPU threads to use
            n_ctx: Context window size
        """
        print("Initializing LLM Query Generator...")
        
        self.logger = self._setup_logger()
        self.model_repo = model_repo
        self.local_dir = Path(local_dir)
        self.model_pattern = model_pattern
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        
        self.llm = None
        self.model_path = None
        
        if not LLM_AVAILABLE:
            print(" LLM libraries not available - query generator will be disabled")
            return
        
        # Initialize the model
        self._initialize_model()
        
        print("LLM Query Generator initialized successfully!")
    
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
    
    def is_available(self) -> bool:
        """Check if LLM query generator is available."""
        return LLM_AVAILABLE and self.llm is not None
    
    def _initialize_model(self) -> bool:
        """
        Initialize the LLM model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model exists locally first
            self.model_path = self._find_local_model()
            
            if not self.model_path:
                print(" Model not found locally, downloading...")
                if not self._download_model():
                    return False
                self.model_path = self._find_local_model()
            
            if not self.model_path:
                self.logger.error("Failed to locate model file")
                return False
            
            print(f"Using model: {self.model_path}")
            
            # Initialize llama-cpp-python
            print(" Loading LLM model...")
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                verbose=False
            )
            
            print("LLM model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM model: {e}")
            return False
    
    def _find_local_model(self) -> Optional[Path]:
        """
        Find the local GGUF model file.
        
        Returns:
            Path to the model file if found, None otherwise
        """
        if not self.local_dir.exists():
            return None
        
        gguf_files = list(self.local_dir.glob("*.gguf"))
        
        if gguf_files:
            # Prefer the specific model pattern
            for file in gguf_files:
                if self.model_pattern in file.name:
                    return file
            # Return the first GGUF file if pattern not found
            return gguf_files[0]
        
        return None
    
    def _download_model(self) -> bool:
        """
        Download the model from Hugging Face.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Enable huggingface_hub transfer acceleration
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            print(f" Downloading model from {self.model_repo}...")
            snapshot_download(
                repo_id=self.model_repo,
                local_dir=str(self.local_dir),
                allow_patterns=[self.model_pattern],
                resume_download=True
            )
            
            print("Model download completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model: {e}")
            return False
    
    def generate_contract_queries(self, contract_type: str = "healthcare") -> List[str]:
        """
        Generate contract analysis queries using the LLM.
        
        Args:
            contract_type: Type of contract to generate queries for
            
        Returns:
            List of generated queries
        """
        if not self.is_available():
            self.logger.warning("LLM not available for query generation")
            return []
        
        try:
            print(f" Generating contract queries for {contract_type} contracts...")
            
            # Create prompt for query generation
            prompt = self._create_query_generation_prompt(contract_type)
            
            # Generate queries
            print(" Running LLM inference...")
            output = self.llm(prompt, max_tokens=500, temperature=0.7)
            
            generated_text = output["choices"][0]["text"].strip()
            print(f"LLM query generation completed")
            print(f"Generated text length: {len(generated_text)} characters")
            
            # Parse the generated queries
            queries = self._parse_generated_queries(generated_text)
            
            print(f" Parsed {len(queries)} queries from LLM output")
            return queries
            
        except Exception as e:
            self.logger.error(f"Failed to generate queries: {e}")
            return []
    
    def _create_query_generation_prompt(self, contract_type: str) -> str:
        """
        Create a prompt for query generation.
        
        Args:
            contract_type: Type of contract
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Generate 5 important contract analysis queries for {contract_type} contracts. 
Each query should focus on a key aspect of contract compliance and terms.

Format: Return each query on a separate line, numbered 1-5.

Topics to consider:
- Timely filing requirements
- Payment terms and fee schedules
- Provider networks and panels
- Confidentiality and privacy
- Termination and renewal clauses

Queries:
"""
        return prompt.strip()
    
    def _parse_generated_queries(self, generated_text: str) -> List[str]:
        """
        Parse queries from LLM-generated text.
        
        Args:
            generated_text: Raw text from LLM
            
        Returns:
            List of parsed queries
        """
        try:
            lines = generated_text.split('\n')
            queries = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    cleaned = line.lstrip('0123456789.-•• ').strip()
                    if len(cleaned) > 10:  # Filter out very short lines
                        queries.append(cleaned)
            
            return queries[:5]  # Return max 5 queries
            
        except Exception as e:
            self.logger.error(f"Failed to parse generated queries: {e}")
            return []
    
    def demonstrate_query_generation(self) -> Dict[str, Any]:
        """
        Demonstrate query generation (for showing LLM capability).
        
        Returns:
            Dictionary containing demonstration results
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "message": "LLM query generator not available"
            }
        
        try:
            print(" Demonstrating LLM query generation...")  
            
            # Generate queries for both TN and WA contracts
            tn_queries = self.generate_contract_queries("Tennessee healthcare")
            wa_queries = self.generate_contract_queries("Washington healthcare")
            
            result = {
                "status": "success",
                "message": "LLM query generation demonstration completed",
                "llm_model": str(self.model_path.name) if self.model_path else "Unknown",
                "generated_queries": {
                    "TN_queries": tn_queries,
                    "WA_queries": wa_queries
                },
                "note": "Generated queries are for demonstration only. Hardcoded queries will be used for actual analysis."
            }
            
            print("Query generation demonstration completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Query generation demonstration failed: {e}")
            return {
                "status": "error",
                "message": f"Demonstration failed: {str(e)}"
            }
    
    def get_hardcoded_queries(self) -> List[str]:
        """
        Get the hardcoded contract queries that will be used for actual analysis.
        
        Returns:
            List of hardcoded queries
        """
        return [
            "Medicaid Timely Filing: This attribute ensures Medicaid claims are submitted and processed within 120 days of service, or from notification of retroactive eligibility, in compliance with regulatory and contractual requirements. Claims must use appropriate service codes and be submitted electronically (EDI) or via NUCC paper forms if needed. Providers must supply all necessary information for payment determination, and respond to additional requests within 60 days or the filing period, whichever is longer. Properly submitted clean claims are adjudicated and paid promptly according to Medicaid program rules and the members health plan.",
            
            "Medicare Timely Filing: This attribute ensures Medicare Advantage claims are submitted and processed within 90 days of service, or from notification of retroactive primary payor responsibility, in compliance with regulatory and Plan requirements. Claims must use appropriate service codes and be submitted electronically (EDI) or via NUCC paper forms if needed. Providers must provide all information necessary for payment determination and respond to additional requests within 60 days or the filing period, whichever is longer. Properly submitted clean claims are adjudicated and paid according to the member's health plan and applicable regulations.",
            
            "No Steerage/SOC: Elevance may create, modify, or discontinue networks, products, and programs based on quality or business needs. Providers participate only in designated networks after meeting credentialing and participation requirements. Certain services may be limited to specific providers in Health Services Designated Networks, and providers must refer members accordingly; reimbursement is not provided for unauthorized services.",
            
            "Medicaid Fee Schedule: The Rate for Covered Services will be calculated as a percentage of the Professional Provider Market Master Fee Schedule in effect on the service date. This schedule may be modified prospectively during the Agreement term. In line with the CRA, reimbursement will not be based on automatic escalators such as Medicare updates or inflation indexes unless approved by Medicaid Affiliate Services. Although Provider is not part of the Medicaid Affiliate Network, if services are rendered to a Medicaid Affiliate's Member, the Provider must accept the applicable state program's Medicaid rate as full payment, subject to regulatory requirements and prior authorization, except for emergencies or self-referred services. Coordination of necessary information between Provider and Medicaid Affiliate will be facilitated upon request.",
            
            "Medicare Fee Schedule: For Members in a Medicare Advantage Network, the Provider will accept as full payment the lesser of Eligible Charges or the Medicare Advantage Rate, minus applicable Cost Shares. The rate is based on the Medicare Fee Schedule and reflects per-claim payments under original Medicare, excluding bonuses or settlements. Payments automatically adjust for CMS directives or regulatory changes. Providers are not compensated for member bad debts, and payments are final based on the system logic and data at claim processing, with no retroactive adjustments."
        ]


if __name__ == "__main__":
    # Test the query generator
    print("Testing LLM Query Generator...")
    
    generator = ContractQueryGenerator()
    
    if generator.is_available():
        # Demonstrate query generation
        demo_result = generator.demonstrate_query_generation()
        print(f"Demo result: {demo_result['status']}")
        
        if demo_result["status"] == "success":
            print("Generated queries preview:")
            for i, query in enumerate(demo_result["generated_queries"]["TN_queries"][:2], 1):
                print(f"{i}. {query[:100]}...")
    else:
        print("LLM query generator not available")
    
    # Show hardcoded queries
    hardcoded = generator.get_hardcoded_queries()
    print(f"\nUsing {len(hardcoded)} hardcoded queries for actual analysis")
