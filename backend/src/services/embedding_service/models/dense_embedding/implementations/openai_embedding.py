from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from ..base import SemanticEmbeddingService
from ......config import Config


class OpenAIEmbeddingService(SemanticEmbeddingService):
    """
    Embedding service that uses OpenAI models via LangChain's OpenAIEmbeddings.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        super().__init__()
        self._model = model
        self._api_key = api_key or Config.OPENAI_API_KEY
        self._embedder: Optional[OpenAIEmbeddings] = None

    def load_model(self) -> None:
        """Initialize the OpenAI embedding model."""
        self._embedder = OpenAIEmbeddings(
            model=self._model,
            api_key=self._api_key,
        )
    
    def _check_model_loaded(self):
        if not self._embedder:
            raise RuntimeError("Embedding model not loaded. Call `load_model()` first.")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a given input text.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._check_model_loaded()
        return self._embedder.embed_query(text)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            An integer representing the dimension size.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._check_model_loaded()
        dummy_vector = self._embedder.embed_query("test")
        return len(dummy_vector)
