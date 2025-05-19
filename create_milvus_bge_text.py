from src.datasets import M3DocVQA
from src.db.milvus import MilvusBgeM3Retriever
from pymilvus import MilvusClient

MILVUS_COLLECTION_NAME = "m3docvqa_text"

milvus_client = MilvusClient(uri="milvus_db/milvus.db")
milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, MILVUS_COLLECTION_NAME)

milvus_bge_m3_retriever.create_collection()
milvus_bge_m3_retriever.create_index()

m3docvqa = M3DocVQA(data_path="parsing/llm_parsing")
m3docvqa.add_text_data_to_milvus_db(MILVUS_COLLECTION_NAME, batch_size=10)