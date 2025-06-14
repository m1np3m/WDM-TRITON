from .base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from typing import List
from numpy import ndarray


class Bgem3Embedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self._model = SentenceTransformer("sentence-transformers/bgem3")

    def embed(self, data: List[str]) -> ndarray:
        return self._model.encode(data, convert_to_numpy=True)
