import time
import json
import os
from pathlib import Path
from loguru import logger
from typing import Any, Dict, List, Union, Optional

from google import genai
from google.genai.types import (
    HarmCategory, 
    HarmBlockThreshold, 
    SafetySetting, 
    GenerateContentConfig
)

from .base import LLMService
from src.config import Config


class GeminiService(LLMService):
    """Google Gemini LLM wrapper with support for text, image, and file generation."""
    
    # Class-level constants
    DEFAULT_MODEL = "gemini-1.5-flash"
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TOP_P = 0.95
    DEFAULT_TOP_K = 40
    FILE_UPLOAD_DELAY = 1  # seconds
    
    _SAFETY_SETTINGS = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        )
    ]
    
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        """Initialize GeminiLLM instance.
        
        Args:
            model: Model name to use
            api_key: Google API key. If not provided, will try to load from environment variable
        """
        super().__init__()
        self._model = model
        self._api_key = api_key or Config.GEMINI_API_KEY
        
        if not self._api_key:
            raise ValueError(f"API key not provided and {self.ENV_API_KEY} environment variable not set")
    
    def _connect(self) -> None:
        """Establish connection to Google Gemini API."""
        self._client = genai.Client(api_key=self._api_key)
    
    def _create_generation_config(
        self, 
        system_prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        json_output: bool = True
    ) -> GenerateContentConfig:
        """Create generation configuration."""
        return GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            response_mime_type="application/json" if json_output else "text/plain",
            safety_settings=self._SAFETY_SETTINGS
        )
    
    def _process_response(self, response_text: str, json_output: bool = True) -> Any:
        """Process and validate response."""
        if not response_text:
            raise Exception("Empty response received")
        
        if json_output:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                raise Exception(f"JSON decode error: {e}")
        
        return response_text
    
    def _create_success_response(self, result: Any) -> Dict[str, Any]:
        """Create successful response dictionary."""
        return {
            "is_success": True,
            "response": result,
            "error": None
        }
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create error response dictionary."""
        return {
            "is_success": False,
            "response": None,
            "error": error
        }
    
    def _upload_file(self, file_path: Union[str, Path], retry: int = DEFAULT_RETRY_COUNT) -> Any:
        """Upload file to Gemini with retry logic."""
        for attempt in range(retry):
            try:
                file = self._client.files.upload(file=file_path)
                logger.debug(f"Uploaded file as: {file.uri}")
                time.sleep(self.FILE_UPLOAD_DELAY)
                return file
            except Exception as e:
                logger.error(f"File upload attempt {attempt + 1} failed: {e}")
                if attempt == retry - 1:
                    raise
                time.sleep(self.FILE_UPLOAD_DELAY * (attempt + 1))
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        file_paths: Optional[List[Union[str, Path]]] = None,
        retry: int = DEFAULT_RETRY_COUNT,
        json_output: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """Generate response with optional file input.
        
        This unified method can handle:
        - Text-only generation (when file_paths is None or empty)
        - Generation with files/images (when file_paths is provided)
        
        Args:
            system_prompt: System instruction
            user_prompt: User input
            file_paths: Optional list of file paths (images, documents, etc.)
            retry: Number of retry attempts
            json_output: Whether to return JSON response
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Response dictionary with success status, response, and error
        """
        try:
            self.ensure_connection()
            
            config = self._create_generation_config(
                system_prompt, temperature, max_output_tokens, top_p, top_k, json_output
            )
            
            if file_paths:
                # Upload files (images, documents, etc.)
                files = [self._upload_file(file_path) for file_path in file_paths]
                
                # Create chat session for file-based generation
                chat_session = self._client.chats.create(
                    model=self._model,
                    config=config
                )
                
                # Send files to context
                for file in files:
                    chat_session.send_message(file)
                
                # Send user prompt
                response = chat_session.send_message(user_prompt)
                result = self._process_response(response.text, json_output)
            else:
                # No files, use regular text generation
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[user_prompt],
                    config=config
                )
                result = self._process_response(response.text, json_output)
            
            return self._create_success_response(result)
            
        except Exception as e:
            if retry <= 0:
                logger.error(f"Generate failed after all retries: {e}")
                return self._create_error_response(str(e))
            
            logger.warning(f"Generate attempt failed, retrying... ({retry} attempts left)")
            return self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                file_paths=file_paths,
                retry=retry - 1,
                json_output=json_output,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k
            )