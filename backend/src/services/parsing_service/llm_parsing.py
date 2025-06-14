import os
import json
from pathlib import Path
from typing import Union, Optional

from loguru import logger
from ..llm_service.factory import LLMServiceFactory


DEFAULT_SYSTEM_PROMPT = """
### ROLE
    - You are a document parsing tool.

### INSTRUCTIONS
    - You will receive a document.
    - Extract all tables from this document, including those that span across multiple pages.
        + Preserve the original row and column structure.
        + If a table continues on the next page(s), merge it properly with the corresponding table segment.
        + If table headers are repeated across pages, remove the duplicated headers in the final output.

### OUTPUT
    - Return the results in JSON format structured as follows:
    [
    {
        "table_id": "table_1",
        "page_range": [2, 3],
        "columns": ["column 1", "column 2", "column 3"],
        "rows": [
        ["row 1 - col 1", "row 1 - col 2", "row 1 - col 3"],
        ["row 2 - col 1", "row 2 - col 2", "row 2 - col 3"]
        ]
    },
    ...
    ]

### REQUIREMENTS
    - Each table should include a unique table_id and a page_range indicating which pages the table spans.
    - If column headers are not clearly defined, try to infer them from the first row of the table.
    - Make sure no tables are skipped, including those without clear borders.
"""


class LLMParsingService:
    """Service for parsing documents using LLM to extract tables."""
    
    # Class constants
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    
    DEFAULT_USER_PROMPT = "Please extract all tables from the document"
    MAX_RETRY_ATTEMPTS = 3
    
    def __init__(
        self, 
        llm_service_name: str = "gemini",
        system_prompt: Optional[str] = None, 
        **kwargs
    ):
        """Initialize the LLM parsing service.
        
        Args:
            llm_service_name: Name of the LLM service to use
            system_prompt: Custom system prompt, uses default if None
            **kwargs: Additional arguments for LLM service
        """
        self.llm_service = LLMServiceFactory.create_service(
            name=llm_service_name, 
            **kwargs
        )
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
    def parse_directory(
        self, 
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        user_prompt: str = DEFAULT_USER_PROMPT
    ) -> None:
        """Parse PDF documents and extract tables.
        
        Args:
            data_dir: Directory containing PDF files to parse
            output_dir: Directory to save parsing results
            user_prompt: Prompt to send to LLM for parsing
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        
        pdf_filenames = self._get_unprocessed_pdfs(data_dir, output_dir)
        logger.info(f"Found {len(pdf_filenames)} PDFs to parse")
        
        for pdf_filename in pdf_filenames:
            self.parse_file(
                pdf_filename=pdf_filename,
                data_dir=data_dir,
                output_dir=output_dir,
                user_prompt=user_prompt
            )
    
    def _get_unprocessed_pdfs(
        self, 
        data_dir: Path, 
        output_dir: Path
    ) -> list[str]:
        """Get list of PDF files that haven't been processed yet.
        
        Args:
            data_dir: Directory containing PDF files
            output_dir: Directory containing output JSON files
            
        Returns:
            List of unprocessed PDF filenames
        """
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return []
            
        all_pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        
        if not output_dir.exists():
            return all_pdfs
            
        existing_outputs = set(os.listdir(output_dir))
        unprocessed_pdfs = [
            pdf_file for pdf_file in all_pdfs 
            if f"{pdf_file}.json" not in existing_outputs
        ]
        
        return unprocessed_pdfs
    
    def parse_file(
        self,
        pdf_filename: Union[str, Path],
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        user_prompt: str = DEFAULT_USER_PROMPT
    ) -> None:
        """Process a single PDF file with retry logic.
        
        Args:
            pdf_filename: Name of the PDF file to process
            data_dir: Directory containing the PDF file
            output_dir: Directory to save the result
            user_prompt: Prompt for LLM parsing
        """
        
        pdf_filename = Path(pdf_filename)
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        
        pdf_path = data_dir / pdf_filename
        
        for attempt in range(1, self.MAX_RETRY_ATTEMPTS + 1):
            try:
                parsing_results = self._parse_with_llm(pdf_path, user_prompt)
                json_result = self._process_parsing_results(parsing_results)
                
                if json_result:
                    self._save_results(pdf_filename, json_result, output_dir)
                    logger.info(f"Successfully parsed {pdf_filename}")
                    break
                else:
                    logger.warning(f"Empty result for {pdf_filename}, attempt {attempt}")
                    
            except Exception as e:
                logger.error(f"Error parsing {pdf_filename} (attempt {attempt}): {e}")
                
            if attempt == self.MAX_RETRY_ATTEMPTS:
                logger.error(f"Failed to parse {pdf_filename} after {attempt} attempts")
    
    def _parse_with_llm(self, pdf_path: Path, user_prompt: str) -> dict:
        """Parse PDF using LLM service.
        
        Args:
            pdf_path: Path to the PDF file
            user_prompt: Prompt for parsing
            
        Returns:
            Parsing results from LLM
        """
        return self.llm_service.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            json_output=True,
            file_paths=[str(pdf_path)]
        )['response']
    
    def _process_parsing_results(self, parsing_results: dict) -> Optional[dict]:
        """Process and validate parsing results.
        
        Args:
            parsing_results: Raw results from LLM
            
        Returns:
            Processed JSON result or None if invalid
        """
        try:
            # Fix the json.dump/json.loads issue
            if isinstance(parsing_results, str):
                return json.loads(parsing_results)
            elif isinstance(parsing_results, dict):
                return parsing_results
            else:
                # Convert to JSON string then back to dict for validation
                json_str = json.dumps(parsing_results)
                return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to process parsing results: {e}")
            return None
    
    def _save_results(
        self, 
        pdf_filename: str, 
        json_result: dict, 
        output_dir: Path
    ) -> None:
        """Save parsing results to JSON file.
        
        Args:
            pdf_filename: Original PDF filename
            json_result: Parsed results to save
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{pdf_filename}.json"
        output_path = output_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)