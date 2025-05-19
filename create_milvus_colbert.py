from src.datasets import M3DocVQA
from src.db.milvus import MilvusColbertRetriever
from pymilvus import MilvusClient

MILVUS_COLLECTION_NAME = "m3docvqa_500"

milvus_client = MilvusClient(uri="milvus_db/milvus.db")
milvus_colbert_retriever = MilvusColbertRetriever(milvus_client, MILVUS_COLLECTION_NAME)

milvus_colbert_retriever.create_collection()
milvus_colbert_retriever.create_index()

m3docvqa = M3DocVQA(data_path="m3docvqa/images_dev", embedding_path="m3docvqa/embeddings_dev")
m3docvqa.add_copali_data_to_milvus_db(MILVUS_COLLECTION_NAME)