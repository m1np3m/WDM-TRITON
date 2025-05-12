import numpy as np
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vidore/colpali-v1.2-hf"

processor = ColPaliProcessor.from_pretrained(model_name)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

def embed_text(text: str) -> torch.Tensor:
    inputs = processor(text=text).to(device)
    with torch.no_grad():
        text_embeds = model(**inputs, return_tensors="pt").embeddings

    return text_embeds

def embed_image(image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image).to(device)
    with torch.no_grad():
        image_embeds = model(**inputs, return_tensors="pt").embeddings

    return image_embeds

    

