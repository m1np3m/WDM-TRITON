from typing import Dict, Type

from ..base import EmbeddingServiceFactory
from ...models.dense_embedding.implementations.bge_m3_embedding import BGEM3EmbeddingService
from ...models.dense_embedding.implementations.colpali_embedding import ColPaliEmbeddingService
from ...models.dense_embedding.implementations.google_embedding import GoogleEmbeddingService
from ...models.dense_embedding.implementations.openai_embedding import OpenAIEmbeddingService
from ...models.dense_embedding.base import DenseEmbeddingService


class DenseEmbeddingServiceFactory(EmbeddingServiceFactory):
    """
    Factory for creating instances of registered semantic embedding services.
    """
    _services: Dict[str, Type[DenseEmbeddingService]] = {
        "bge_m3": BGEM3EmbeddingService,
        "colpali": ColPaliEmbeddingService,
        "google": GoogleEmbeddingService,
        "openai": OpenAIEmbeddingService
    }