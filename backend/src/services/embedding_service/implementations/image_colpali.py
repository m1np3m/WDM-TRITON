from .base import BaseEmbedding
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
from PIL import Image
from typing import List, Union
from pathlib import Path
import numpy as np
from numpy import ndarray
from loguru import logger


class ImageColpaliEmbedding(BaseEmbedding):
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

    def _process_image(
        self, image_list: List[Union[str, Path, Image.Image]]
    ) -> torch.Tensor:
        inputs = []
        for image in image_list:
            if isinstance(image, str) or isinstance(image, Path):
                img_path = Path(image)
                if not img_path.exists():
                    raise FileNotFoundError(f"File not found: {img_path}")
                image = Image.open(img_path)
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            input = self.processor(images=image).to(self.device)
            inputs.append(input)
        return torch.stack(inputs).to(self.device)

    def embed(
        self, data: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]
    ) -> ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model()")

        image_list = [data] if not isinstance(data, list) else data

        try:
            with torch.no_grad():
                inputs = self._process_image(image_list)
                return (
                    self.model.forward(inputs, return_tensors="pt")
                    .embeddings.cpu()
                    .numpy()
                    .astype(np.float64)
                )
        except Exception as e:
            logger.error(f"Failed to embed data with model {self.model_name}: {e}")
            raise
