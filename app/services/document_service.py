import os
import time
from pathlib import Path
from typing import Optional, Dict
import pymupdf

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("document_service")


class DocumentService:
    """
    Handles PDF inspection, OCR preprocessing, text extraction, and multi-tenant file management.
    """

    def __init__(self, output_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.processed_dir = processed_dir or settings.PROCESSED_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

    def get_user_doc_dir(self, user_id: int) -> str:
        path = os.path.join(settings.DOCUMENTS_DIR, f"user_{user_id}")
        os.makedirs(path, exist_ok=True)
        return path

    def is_searchable_pdf(self, pdf_path: str, min_chars_per_page: int = 20) -> bool:
        """
        Check whether a PDF contains a searchable text layer.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        searchable_pages = 0

        for page in doc:
            text = page.get_text().strip()
            if len(text) >= min_chars_per_page:
                searchable_pages += 1

        doc.close()

        ratio = (searchable_pages / total_pages) if total_pages > 0 else 0.0
        logger.info(
            f"PDF searchability check for {pdf_path}: {searchable_pages}/{total_pages} pages ({ratio:.2%})"
        )
        return ratio >= 0.5

    def run_ocr(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Run OCRmyPDF on a scanned or non-searchable PDF.
        """
        if not output_path:
            base_name = Path(pdf_path).stem
            output_path = os.path.join(self.processed_dir, f"{base_name}_ocr.pdf")

        logger.info(f"Starting OCR for {pdf_path} -> {output_path}")
        start_time = time.time()

        try:
            import ocrmypdf
            ocrmypdf.ocr(
                pdf_path,
                output_path,
                skip_text=True,
                deskew=False,
                clean=False,
                optimize=0,
                language="eng",
            )
            elapsed = time.time() - start_time
            logger.info(f"OCR completed in {elapsed:.2f} seconds")
            return output_path
        except Exception as e:
            logger.warning(f"OCRmyPDF unavailable or failed ({e}). Returning original PDF.")
            return pdf_path

    def prepare_pdf(self, pdf_path: str) -> str:
        """
        Ensures the PDF is searchable by running OCR if needed.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.is_searchable_pdf(pdf_path):
            logger.info(f"PDF is searchable: {pdf_path}")
            return pdf_path
        else:
            logger.info(f"PDF is non-searchable. Triggering OCR for {pdf_path}")
            return self.run_ocr(pdf_path)

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF using high-performance PyMuPDF streaming with fallback.
        """
        import gc
        searchable_pdf = self.prepare_pdf(pdf_path)
        logger.info(f"Extracting text from: {searchable_pdf} into {self.output_dir}")

        base_name = Path(searchable_pdf).stem
        txt_file_path = os.path.join(self.output_dir, f"{base_name}.txt")

        text_parts = []
        try:
            doc = pymupdf.open(searchable_pdf)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
            doc.close()
            text = "\n\n".join(text_parts)
        except Exception as ex:
            logger.warning(f"PyMuPDF direct extraction note: {ex}. Trying opendataloader_pdf...")
            try:
                import opendataloader_pdf
                opendataloader_pdf.convert(
                    input_path=[searchable_pdf],
                    output_dir=self.output_dir,
                    format="text"
                )
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    raise ex
            except Exception as e2:
                logger.error(f"Fallback text extraction failed: {e2}")
                raise ex

        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        del text_parts
        gc.collect()

        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text

    def delete_document_files(self, file_path: str, filename: str) -> Dict[str, bool]:
        """
        Deletes the physical raw PDF, processed OCR PDF, and extracted TXT files.
        """
        base_name = Path(filename).stem
        deleted = {}

        # 1. Raw PDF
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted["raw_pdf"] = True
                logger.info(f"Deleted raw PDF: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                deleted["raw_pdf"] = False

        # 2. Processed OCR PDF
        ocr_pdf = os.path.join(self.processed_dir, f"{base_name}_ocr.pdf")
        if os.path.exists(ocr_pdf):
            try:
                os.remove(ocr_pdf)
                deleted["ocr_pdf"] = True
                logger.info(f"Deleted OCR PDF: {ocr_pdf}")
            except Exception as e:
                logger.error(f"Failed to delete {ocr_pdf}: {e}")
                deleted["ocr_pdf"] = False

        # 3. Extracted TXT
        extracted_txt = os.path.join(self.output_dir, f"{base_name}.txt")
        if os.path.exists(extracted_txt):
            try:
                os.remove(extracted_txt)
                deleted["extracted_txt"] = True
                logger.info(f"Deleted extracted TXT: {extracted_txt}")
            except Exception as e:
                logger.error(f"Failed to delete {extracted_txt}: {e}")
                deleted["extracted_txt"] = False

        return deleted
