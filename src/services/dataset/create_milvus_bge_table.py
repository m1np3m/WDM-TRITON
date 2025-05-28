import os
import sys

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.datasets import M3DocVQA
from src.db.milvus import MilvusBgeM3Retriever
from pymilvus import MilvusClient

MILVUS_TABLE_COLLECTION_NAME = "m3docvqa_table"
milvus_client = MilvusClient(uri="milvus_db/milvus.db")

milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, MILVUS_TABLE_COLLECTION_NAME)

milvus_bge_m3_retriever.create_collection()
milvus_bge_m3_retriever.create_index()

m3docvqa = M3DocVQA(data_path="m3docvqa/tables_description")
m3docvqa.add_table_data_to_milvus_db(MILVUS_TABLE_COLLECTION_NAME, "m3docvqa/tables_dev", batch_size=2)