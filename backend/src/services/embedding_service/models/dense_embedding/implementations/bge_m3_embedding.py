from typing import List, Optional

import torch
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from ..base import DenseEmbeddingService
from ......config import Config


class BGEM3EmbeddingService(DenseEmbeddingService):
    """
    Service for generating text embeddings using the BGE-M3 model from pymilvus.
    """

    def __init__(self, model: str = "BAAI/bge-m3", device: Optional[str] = None):
        super().__init__()
        self._model = model
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embedder: Optional[BGEM3EmbeddingFunction] = None

    def load_model(self) -> None:
        """Load the BGE-M3 embedding model."""
        self._embedder = BGEM3EmbeddingFunction(
            model_name=self._model,
            device=self._device,
            use_fp16=False
        )

    def _check_model_loaded(self) -> None:
        if not self._embedder:
            raise RuntimeError("Model not loaded. Call `load_model()` first.")

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        self._check_model_loaded()
        return self._embedder(text)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: Size of the embedding vector.
        """
        self._check_model_loaded()
        return len(self._embedder("test"))
