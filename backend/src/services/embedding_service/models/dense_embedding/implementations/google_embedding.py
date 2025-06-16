from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..base import DenseEmbeddingService
from ......config import Config


class GoogleEmbeddingService(DenseEmbeddingService):
    """
    Embedding service that uses Google's Generative AI via LangChain.
    """

    def __init__(self, model: str = "gemini-embedding-001", api_key: Optional[str] = None):
        super().__init__()
        self._model = model
        self._api_key = api_key or Config.GEMINI_API_KEY
        self._embedder: Optional[GoogleGenerativeAIEmbeddings] = None

    def load_model(self) -> None:
        """Initialize the Google Generative AI embedding model."""
        self._embedder = GoogleGenerativeAIEmbeddings(
            model=self._model,
            api_key=self._api_key
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
            An integer representing the embedding dimension.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._check_model_loaded()
        dummy_vector = self._embedder.embed_query("test")
        return len(dummy_vector)
