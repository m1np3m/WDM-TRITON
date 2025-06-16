from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

from ..base import SemanticEmbeddingService


class ColPaliEmbeddingService(SemanticEmbeddingService):
    """
    Embedding service using the ColPali model for both text and image inputs.
    """

    def __init__(self, model: str = "vidore/colpali-v1.2-hf", device: Optional[str] = None):
        self._model = model
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embedder: Optional[ColPaliForRetrieval] = None
        self._processor: Optional[ColPaliProcessor] = None

    def load_model(self) -> None:
        """Loads the ColPali model and processor."""
        self._embedder = ColPaliForRetrieval.from_pretrained(
            self._model,
            torch_dtype=torch.bfloat16,
            device_map=self._device
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(self._model)

    def _check_model_loaded(self):
        if not self._embedder or not self._processor:
            raise RuntimeError("Model not loaded. Call `load_model()` first.")

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate a text embedding.

        Args:
            text: Input text string.

        Returns:
            List of float values representing the embedding.
        """
        self._check_model_loaded()
        inputs = self._processor(text=text, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            embeddings = self._embedder(**inputs).embeddings

        return embeddings.squeeze(0).cpu().numpy().tolist()

    def get_image_embedding(self, image: Union[str, Path, Any]) -> List[float]:
        """
        Generate an image embedding.

        Args:
            image: Image input (file path, PIL image, or numpy array)

        Returns:
            List of float values representing the embedding.
        """
        self._check_model_loaded()
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            embeddings = self._embedder(**inputs).embeddings

        return embeddings.squeeze(0).cpu().numpy().tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vector.

        Returns:
            Integer representing the embedding vector size.
        """
        self._check_model_loaded()
        inputs = self._processor(text="test", return_tensors="pt").to(self._device)

        with torch.inference_mode():
            embeddings = self._embedder(**inputs).embeddings

        return embeddings.shape[-1]
