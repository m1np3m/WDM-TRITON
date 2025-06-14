from .implementations.bgem3 import Bgem3Embedding
from .implementations.text_colpali import TextColpaliEmbedding
from .implementations.image_colpali import ImageColpaliEmbedding
from .implementations.base import BaseEmbedding


def get_embedding_service(service_name: str) -> BaseEmbedding:
    if service_name == "bgem3":
        return Bgem3Embedding()
    elif service_name == "text_colpali":
        return TextColpaliEmbedding()
    elif service_name == "image_colpali":
        return ImageColpaliEmbedding()
    else:
        raise ValueError(f"Unknown embedding service: {service_name}")
