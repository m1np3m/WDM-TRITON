import torch
import json
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from dataclasses import dataclass

from loguru import logger
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


@dataclass
class ParseResult:
    """Parsing result container."""
    file_path: str
    markdown: Optional[str] = None
    doctags: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file_path, "markdown": self.markdown, 
                "doctags": self.doctags, "error": self.error}


class DoclingParsingService:
    """Document parsing service using Docling."""
    
    MODEL_NAME = "ds4sd/SmolDocling-256M-preview"
    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.pdf'}
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.bfloat16, _attn_implementation="eager"
        ).to(self.device)
        logger.info(f"Model loaded on {self.device}")
    
    def parse_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, int]:
        """Parse all supported files in directory."""
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = self._get_files(input_dir)
        stats = {"total": len(files), "processed": 0, "failed": 0}
        
        logger.info(f"Processing {len(files)} files")
        
        for file_path in files:
            output_file = output_dir / f"{file_path.stem}.json"
            if output_file.exists():
                continue
                
            result = self.parse_file(file_path)
            self._save_result(result, output_file)
            
            if result.error:
                stats["failed"] += 1
            else:
                stats["processed"] += 1
        
        logger.info(f"Completed: {stats}")
        return stats
    
    def parse_file(self, file_path: Union[str, Path], prompt: str = "Parse this document") -> ParseResult:
        """Parse single file."""
        file_path = Path(file_path)
        
        try:
            image = load_image(str(file_path))
            doctags = self._generate_doctags(image, prompt)
            markdown = self._to_markdown(doctags, image, file_path)
            
            return ParseResult(str(file_path), markdown, doctags)
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return ParseResult(str(file_path), error=str(e))
    
    def _get_files(self, directory: Path) -> List[Path]:
        """Get all supported files."""
        files = []
        for ext in self.SUPPORTED_EXTS:
            files.extend(directory.glob(f"*{ext}"))
        return sorted(files)
    
    def _generate_doctags(self, image, prompt: str) -> str:
        """Generate doctags using model."""
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        
        chat_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat_prompt, images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        
        prompt_length = inputs.input_ids.shape[1]
        return self.processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=False)[0].lstrip()
    
    def _to_markdown(self, doctags: str, image, file_path: Path) -> str:
        """Convert doctags to markdown."""
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name=str(file_path))
        doc.load_from_doctags(doctags_doc)
        return doc.export_to_markdown()
    
    def _save_result(self, result: ParseResult, output_path: Path) -> None:
        """Save result to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=4, ensure_ascii=False)