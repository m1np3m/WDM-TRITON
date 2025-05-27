from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch
from typing import List

class BgeM3MilvusEmbedding:
    def __init__(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3EmbeddingFunction(device=device)
        
    def encode(self, texts: List[str]):
        return self.model(texts)