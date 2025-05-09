from src.datasets import M3DocVQA
from src.db import chroma_client

CHROME_COLLECTION_NAME = "m3docvqa_500"

m3docvqa = M3DocVQA(image_path="m3docvqa/images_dev", embedding_path="m3docvqa/embeddings_dev")
m3docvqa.add_data_to_db(CHROME_COLLECTION_NAME, batch_size=100)
