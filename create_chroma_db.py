from src.datasets import M3DocVQA
from src.db import chroma_client

CHROMA_COLLECTION_NAME = "m3docvqa_500"

m3docvqa = M3DocVQA(image_path="m3docvqa/images_dev", embedding_path="m3docvqa/embeddings_dev")

chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
chroma_client.create_collection(CHROMA_COLLECTION_NAME)
m3docvqa.add_data_to_chroma_db(CHROMA_COLLECTION_NAME, batch_size=100)
