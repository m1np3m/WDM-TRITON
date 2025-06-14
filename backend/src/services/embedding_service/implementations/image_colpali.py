from .base import BaseEmbedding
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
from PIL import Image
from numpy import ndarray

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vidore/colpali-v1.2-hf"


class ImageColpaliEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self._model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(model_name)

    def embed(self, data: list[Image.Image]) -> ndarray:
        inputs = self._processor(images=data).to(device)
        with torch.no_grad():
            image_embeds = self._model(**inputs, return_tensors="pt").embeddings

        return image_embeds.cpu().numpy()
