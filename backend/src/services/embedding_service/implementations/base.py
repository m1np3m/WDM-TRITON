from abc import ABC, abstractmethod
from typing import List, Any, Union
from numpy import ndarray


class BaseEmbedding(ABC):
    def __init__(self) -> None:
        self.model_name = None
        self.pretrained = None
        self.device = None
        self.model = None
        self.tokenizer = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def embed(self, data: Union[Any, List[Any]]) -> ndarray:
        pass
