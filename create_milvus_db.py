from src.datasets import M3DocVQA
from src.db.milvus import milvus_colbert_retriever

MILVUS_COLLECTION_NAME = "m3docvqa_500"

milvus_colbert_retriever.create_collection()
milvus_colbert_retriever.create_index()

m3docvqa = M3DocVQA(image_path="m3docvqa/images_dev", embedding_path="m3docvqa/embeddings_dev")
m3docvqa.add_data_to_milvus_db(MILVUS_COLLECTION_NAME)