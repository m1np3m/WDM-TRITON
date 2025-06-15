from typing import Dict, Type

from ..base import EmbeddingServiceFactory
from ...models.semantic_embedding.implementations.bge_m3_embedding import BGEM3EmbeddingService
from ...models.semantic_embedding.implementations.colpali_embedding import ColPaliEmbeddingService
from ...models.semantic_embedding.implementations.google_embedding import GoogleEmbeddingService
from ...models.semantic_embedding.implementations.openai_embedding import OpenAIEmbeddingService
from ...models.semantic_embedding.base import SemanticEmbeddingService


class SemanticEmbeddingServiceFactory(EmbeddingServiceFactory):
    """
    Factory for creating instances of registered semantic embedding services.
    """
    _services: Dict[str, Type[SemanticEmbeddingService]] = {
        "bge_m3": BGEM3EmbeddingService,
        "colpali": ColPaliEmbeddingService,
        "google": GoogleEmbeddingService,
        "openai": OpenAIEmbeddingService
    }