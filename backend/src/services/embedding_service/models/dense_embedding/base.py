from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

from ..base import EmbeddingService

class DenseEmbeddingService(EmbeddingService, ABC):
    """
    Abstract base class for dense embedding services.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
        pass

    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors."""
        pass

    def get_image_embedding(self, image: Union[str, Path, Any]) -> List[float]:
        """Get embedding for an image."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support image embeddings.")

    def get_batch_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        return [self.get_text_embedding(text) for text in texts]

    def get_batch_image_embeddings(self, images: List[Union[str, Path, Any]]) -> List[List[float]]:
        """Get embeddings for multiple images efficiently."""
        if not self.supports_image_embedding():
            raise NotImplementedError(f"{self.__class__.__name__} does not support image embeddings.")
        return [self.get_image_embedding(image) for image in images]

    def supports_image_embedding(self) -> bool:
        """Check if the service supports image embeddings."""
        # Considered supported only if `get_image_embedding` is overridden
        return self.__class__.get_image_embedding is not DenseEmbeddingService.get_image_embedding
