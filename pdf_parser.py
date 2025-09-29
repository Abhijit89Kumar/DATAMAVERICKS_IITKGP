#!/usr/bin/env python3
"""
PDF Parser Module - Robust PDF Parsing with OCR and Table Extraction
====================================================================

This module contains the RobustPDFParser class for comprehensive PDF parsing.

Features:
- Multi-page table extraction
- OCR for image-based content
- Markdown conversion
- Support for various PDF types (text-based, image-based, mixed)

Author: AI Assistant
"""

import os
import sys
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings("ignore")

# Core PDF processing imports
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

# PDF and document processing (optional)
try:
    from docling.document_converter import DocumentConverter
    try:
        from docling.datamodel.base_models import ConversionResult
    except ImportError:
        try:
            from docling.datamodel.document import ConversionResult
        except ImportError:
            ConversionResult = None
    DOCLING_AVAILABLE = True
    print("Docling available for advanced PDF processing")
except ImportError as e:
    DOCLING_AVAILABLE = False
    ConversionResult = None
    print(f"Docling not available: {e}")

try:
    import camelot
    CAMELOT_AVAILABLE = True
    print("Camelot available for table extraction")
except ImportError:
    CAMELOT_AVAILABLE = False
    print("Camelot not available - table extraction may be limited")

# OCR libraries (optional)
try:
    import pytesseract
    import cv2
    from PIL import Image
    OCR_AVAILABLE = True
    print("OCR libraries available")
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available")


class RobustPDFParser:
    """
    A robust PDF parser that combines multiple tools for comprehensive document extraction.
    """

    def __init__(self, enable_ocr=True, ocr_language='en'):
        """
        Initialize the PDF parser.

        Args:
            enable_ocr (bool): Enable OCR for image-based content
            ocr_language (str): Language for OCR (default: 'en')
        """
        print("Initializing PDF Parser...")
        
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        self.setup_logging()

        # Initialize EasyOCR if available
        if self.enable_ocr and EASYOCR_AVAILABLE:
            try:
                print("Loading EasyOCR...")
                self.easyocr_reader = easyocr.Reader([ocr_language])
                print("EasyOCR initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize EasyOCR: {e}")
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None

        # Initialize docling if available
        if DOCLING_AVAILABLE:
            try:
                print("Loading Docling converter...")
                self.docling_converter = DocumentConverter()
                print("Docling converter initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Docling: {e}")
                self.docling_converter = None
        else:
            self.docling_converter = None

        print("PDF Parser initialized successfully")

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_with_docling(self, pdf_path: str) -> Optional[str]:
        """
        Extract text using docling (primary method).

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Optional[str]: Extracted text in markdown format
        """
        if not DOCLING_AVAILABLE or not self.docling_converter:
            self.logger.info("Docling not available, skipping")
            return None

        try:
            self.logger.info(f"Extracting text with docling from: {Path(pdf_path).name}")
            result = self.docling_converter.convert(pdf_path)

            # Try different methods to get markdown content
            try:
                content = result.document.export_to_markdown()
                self.logger.info("Docling extraction successful")
                return content
            except AttributeError:
                try:
                    content = result.markdown
                    self.logger.info("Docling extraction successful (alternative method)")
                    return content
                except AttributeError:
                    try:
                        content = str(result.document)
                        self.logger.info("Docling extraction successful (fallback method)")
                        return content
                    except AttributeError:
                        content = str(result)
                        self.logger.info("Docling extraction successful (string conversion)")
                        return content

        except Exception as e:
            self.logger.error(f"Docling extraction failed: {e}")
            return None

    def extract_tables_with_camelot(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using camelot (specialized table extraction).

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[pd.DataFrame]: List of extracted tables
        """
        tables = []

        if not CAMELOT_AVAILABLE:
            self.logger.info("⏭ Camelot not available, skipping table extraction")
            return tables

        try:
            self.logger.info(f"Extracting tables with camelot from: {Path(pdf_path).name}")

            # Try lattice method first (for tables with lines)
            try:
                camelot_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
                for i, table in enumerate(camelot_tables):
                    if hasattr(table, 'df') and not table.df.empty:
                        tables.append(table.df)
                        self.logger.info(f"Extracted table {i+1} with lattice method: {table.df.shape}")
            except Exception as e:
                self.logger.warning(f"Lattice method failed: {e}")

            # Try stream method (for tables without lines)
            if not tables:  # Only try stream if lattice didn't work
                try:
                    camelot_tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
                    for i, table in enumerate(camelot_tables):
                        if hasattr(table, 'df') and not table.df.empty:
                            tables.append(table.df)
                            self.logger.info(f"Extracted table {i+1} with stream method: {table.df.shape}")
                except Exception as e:
                    self.logger.warning(f"Stream method failed: {e}")

        except Exception as e:
            self.logger.error(f"Camelot table extraction failed: {e}")

        return tables

    def extract_tables_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables using pdfplumber (fallback method).

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[pd.DataFrame]: List of extracted tables
        """
        tables = []
        try:
            self.logger.info(f"Extracting tables with pdfplumber from: {Path(pdf_path).name}")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables()
                        for table_num, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                tables.append(df)
                                self.logger.info(f"Extracted table from page {page_num + 1}, table {table_num + 1}: {df.shape}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract tables from page {page_num + 1}: {e}")
        except Exception as e:
            self.logger.error(f"PDFPlumber table extraction failed: {e}")

        return tables

    def extract_images_and_ocr(self, pdf_path: str) -> List[str]:
        """
        Extract images from PDF and apply OCR.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[str]: List of OCR text from images
        """
        if not self.enable_ocr:
            self.logger.info("⏭ OCR disabled, skipping image extraction")
            return []

        ocr_texts = []
        try:
            self.logger.info(f" Extracting images and applying OCR to: {Path(pdf_path).name}")
            doc = fitz.open(pdf_path)

            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]

                    # Get page as image for OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    img_data = pix.tobytes("png")

                    # Apply OCR using EasyOCR (preferred) or Tesseract
                    if self.easyocr_reader:
                        try:
                            results = self.easyocr_reader.readtext(img_data)
                            page_text = " ".join([result[1] for result in results])
                            if page_text.strip():
                                ocr_texts.append(f"## OCR Text from Page {page_num + 1}\n\n{page_text}\n")
                                self.logger.info(f"OCR completed for page {page_num + 1} (EasyOCR)")
                        except Exception as e:
                            self.logger.warning(f"EasyOCR failed on page {page_num + 1}: {e}")
                    else:
                        # Fallback to Tesseract
                        try:
                            image = Image.open(io.BytesIO(img_data))
                            page_text = pytesseract.image_to_string(image, lang=self.ocr_language)
                            if page_text.strip():
                                ocr_texts.append(f"## OCR Text from Page {page_num + 1}\n\n{page_text}\n")
                                self.logger.info(f"OCR completed for page {page_num + 1} (Tesseract)")
                        except Exception as e:
                            self.logger.warning(f"Tesseract OCR failed on page {page_num + 1}: {e}")

                except Exception as e:
                    self.logger.warning(f"Failed to process page {page_num + 1} for OCR: {e}")

            doc.close()

        except Exception as e:
            self.logger.error(f"Image extraction and OCR failed: {e}")

        return ocr_texts

    def extract_basic_text(self, pdf_path: str) -> str:
        """
        Extract basic text using PyMuPDF (fallback method).

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            str: Extracted text
        """
        text = ""
        try:
            self.logger.info(f"Extracting basic text from: {Path(pdf_path).name}")
            doc = fitz.open(pdf_path)

            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text += f"## Page {page_num + 1}\n\n{page_text}\n\n"
                        self.logger.info(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

            doc.close()
            self.logger.info("Basic text extraction completed")

        except Exception as e:
            self.logger.error(f"Basic text extraction failed: {e}")

        return text

    def tables_to_markdown(self, tables: List[pd.DataFrame]) -> str:
        """
        Convert extracted tables to markdown format.

        Args:
            tables (List[pd.DataFrame]): List of tables

        Returns:
            str: Tables in markdown format
        """
        markdown_content = ""

        for i, table in enumerate(tables):
            try:
                # Clean up table data
                table = table.fillna("")
                table = table.astype(str)

                # Convert to markdown
                table_md = table.to_markdown(index=False)
                markdown_content += f"\n## Table {i + 1}\n\n{table_md}\n\n"
                self.logger.info(f"Converted table {i + 1} to markdown")

            except Exception as e:
                self.logger.error(f"Failed to convert table {i + 1} to markdown: {e}")

        return markdown_content

    def parse_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Main method to parse PDF with all available methods.

        Args:
            pdf_path (str): Path to PDF file
            output_path (Optional[str]): Path to save markdown output

        Returns:
            str: Complete markdown content
        """
        self.logger.info(f"Starting comprehensive PDF parsing: {Path(pdf_path).name}")
        print(f"\n{'='*70}")
        print(f"PROCESSING PDF: {Path(pdf_path).name}")
        print(f"{'='*70}")

        markdown_content = f"# Document: {Path(pdf_path).name}\n\n"
        markdown_content += "---\n\n"

        # Method 1: Try docling first (primary method)
        print(" STEP 1: Advanced text extraction with Docling...")
        docling_text = self.extract_text_with_docling(pdf_path)
        if docling_text:
            markdown_content += "## Document Content (via Docling)\n\n"
            markdown_content += docling_text + "\n\n"
            markdown_content += "---\n\n"
            print("Docling extraction successful")
        else:
            print("Docling extraction failed, will use fallback methods")

        # # Method 2: Extract tables with specialized tools (commented out in original)
        # print("\n STEP 2: Table extraction...")
        # tables = []

        # # Try camelot first
        # camelot_tables = self.extract_tables_with_camelot(pdf_path)
        # tables.extend(camelot_tables)

        # # Try pdfplumber as fallback
        # if not camelot_tables:
        #     pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
        #     tables.extend(pdfplumber_tables)

        # if tables:
        #     markdown_content += "## Extracted Tables\n\n"
        #     markdown_content += self.tables_to_markdown(tables)
        #     markdown_content += "---\n\n"
        #     print(f"Extracted {len(tables)} tables successfully")
        # else:
        #     print("ℹ No tables found in the document")

        # # Method 3: OCR for image-based content
        # print("\n STEP 3: OCR processing...")
        # if self.enable_ocr:
        #     ocr_texts = self.extract_images_and_ocr(pdf_path)
        #     if ocr_texts:
        #         markdown_content += "## OCR Content from Images\n\n"
        #         markdown_content += "\n".join(ocr_texts)
        #         markdown_content += "---\n\n"
        #         print(f"OCR processing completed for {len(ocr_texts)} pages")
        #     else:
        #         print("ℹ No OCR content extracted")
        # else:
        #     print("⏭ OCR disabled")

        # Method 4: Basic text extraction as ultimate fallback
        print("\n STEP 4: Fallback text extraction...")
        if not docling_text:
            basic_text = self.extract_basic_text(pdf_path)
            if basic_text:
                markdown_content += "## Basic Text Content\n\n"
                markdown_content += basic_text
                markdown_content += "\n---\n\n"
                print("Fallback text extraction successful")
            else:
                print("All text extraction methods failed")
        else:
            print("Skipping fallback (Docling was successful)")

        # Save markdown file if output path is provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                self.logger.info(f"Markdown saved to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save markdown: {e}")

        print(f"\nPDF parsing completed successfully!")
        print(f"Total content length: {len(markdown_content)} characters")
        
        return markdown_content


if __name__ == "__main__":
    # Test the parser with a sample file
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        parser = RobustPDFParser()
        result = parser.parse_pdf(pdf_path)
        print(f"\nParsing completed. Content length: {len(result)} characters")
    else:
        print("Usage: python pdf_parser.py <pdf_path>")
