from .base import BaseEmbedding
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
import numpy as np
from numpy import ndarray
from loguru import logger


class TextColpaliEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device

    def load_model(self) -> None:
        try:
            self.model = ColPaliForRetrieval.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def embed(self, data: list[str]) -> ndarray:
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before embedding data."
            )
        if not isinstance(data, (str, list)):
            raise TypeError("Data must be a string or a list of strings.")
        if not data:
            raise ValueError("Data must not be empty.")

        text_list = [data] if isinstance(data, str) else data

        try:
            with torch.no_grad():
                inputs = self.processor(text=text_list, return_tensors="pt").to(
                    self.device
                )
                return (
                    self.model.forward(**inputs, return_tensors="pt")
                    .embeddings.cpu()
                    .numpy()
                    .astype(np.float64)
                )
        except Exception as e:
            logger.error(f"Failed to embed data with model {self.model_name}: {e}")
            raise
