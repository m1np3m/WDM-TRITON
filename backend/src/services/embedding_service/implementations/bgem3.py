from .base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from typing import List, Union
from numpy import ndarray
import torch
from loguru import logger


class Bgem3Embedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "sentence-transformers/bgem3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device

    def load_model(self) -> None:
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def embed(self, data: Union[str, List[str]]) -> ndarray:
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before embedding data."
            )
        if not isinstance(data, (str, list)):
            raise TypeError("Data must be a string or a list of strings.")
        if not data:
            raise ValueError("Data must not be empty.")

        data = [data] if isinstance(data, str) else data

        try:
            return self.model.encode(data, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to embed data with model {self.model_name}: {e}")
            raise
