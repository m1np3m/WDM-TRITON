import os
import sys
from pathlib import Path

# Add root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from src.datasets import M3DocVQA
from src.db.milvus import MilvusBgeM3Retriever, MilvusLLMRetriever, MilvusColbertRetriever
from src.db.chroma_client import ChromaRetriever
from pymilvus import MilvusClient
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="parsing/llm_parsing/")
    parser.add_argument("--table_path", type=str)
    parser.add_argument("--type", type=str, default="text")
    parser.add_argument("--db", type=str, default="milvus")
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--embedding_type", type=str, default="llm")
    parser.add_argument("--collection_name", type=str, default="m3docvqa_text_llm")
    parser.add_argument("--batch_size", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    
    embedding_type = args.embedding_type
    collection_name = args.collection_name
    batch_size = args.batch_size

    m3docvqa = M3DocVQA(args.dataset_path, args.embedding_path)

    if args.type == "multimodal_copali":
        if args.db == "milvus":
            milvus_client = MilvusClient(uri="milvus_db/milvus.db")
            milvus_colbert_retriever = MilvusColbertRetriever(milvus_client, collection_name)
            milvus_colbert_retriever.create_collection()
            milvus_colbert_retriever.create_index()
            m3docvqa.add_copali_data_to_milvus_db(collection_name, batch_size = 1)

        elif args.db == "chroma":
            chroma_retriever = ChromaRetriever(collection_name)
            chroma_retriever.create_collection()
            m3docvqa.add_copali_data_to_chroma_db(collection_name, batch_size = 1)

    elif args.type == "table":
        milvus_client = MilvusClient(uri="milvus_db/milvus.db")
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            milvus_bge_m3_retriever.create_collection()
            milvus_bge_m3_retriever.create_index()
            m3docvqa.add_table_data_to_milvus_db(collection_name, args.table_path, batch_size, embedding_type)
        
        elif embedding_type == "llm":
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            milvus_llm_retriever.create_collection()
            milvus_llm_retriever.create_index()
            m3docvqa.add_table_data_to_milvus_db(collection_name, args.table_path, batch_size, embedding_type)
    
    elif args.type == "text":
        milvus_client = MilvusClient(uri="milvus_db/milvus.db")
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            milvus_bge_m3_retriever.create_collection()
            milvus_bge_m3_retriever.create_index()
            m3docvqa.add_text_data_to_milvus_db(collection_name, batch_size, embedding_type)
        elif embedding_type == "llm":
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            milvus_llm_retriever.create_collection()
            milvus_llm_retriever.create_index()
            m3docvqa.add_text_data_to_milvus_db(collection_name, batch_size, embedding_type)
            
if __name__ == "__main__":
    main()
