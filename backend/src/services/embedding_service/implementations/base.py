from abc import ABC, abstractmethod
from typing import List, Any
from numpy import ndarray


class BaseEmbedding(ABC):
    def __init__(self):
        self._model = None
        self._processor = None

    @abstractmethod
    def embed(self, data: List[Any]) -> ndarray:
        pass
