import glob
import json
import os
import numpy as np
from src.datasets.dataset import Dataset
from src.db.chroma_client import ChromaRetriever
from src.db.milvus import MilvusColbertRetriever, MilvusBgeM3Retriever, MilvusLLMRetriever
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models.embedding.bge_m3 import BgeM3MilvusEmbedding
from src.models.embedding.llm import LLMEmbedding
from src.services.processing.text_processing import merge_json_data, processing_dict_data
import time

milvus_client = MilvusClient(uri="milvus_db/milvus.db")

class M3DocVQA(Dataset):
    def __init__(self, data_path: str, embedding_path: str=None):
        super().__init__(data_path, embedding_path)

    def load_data(self):
        self.data = []

    def add_copali_data_to_chroma_db(self, collection_name: str, batch_size: int = 100):
        image_files = glob.glob(os.path.join(self.data_path, "*"))
        chrome_retriever = ChromaRetriever(collection_name)
        # Process files in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_data = []
            
            img_num = 0
            for image_file in batch_files:
                print(image_file)
                image_name = os.path.basename(image_file)
                embedding_file = os.path.join(self.embedding_path, image_name + ".npy")

                embedding = np.load(embedding_file)
                embedding = np.squeeze(embedding, axis=0)
                aggregated_embedding = np.mean(embedding, axis=0)
                aggregated_embedding = aggregated_embedding.tolist()

                batch_data.append({"id": f"batch_{i}_{img_num}", "image_name": image_name, "embedding": aggregated_embedding})
                img_num += 1

            chrome_retriever.add_image_embeddings(batch_data)

    def get_data(self):
        return self.data

    def add_copali_data_to_milvus_db(self, collection_name: str, batch_size: int = 1):
        milvus_colbert_retriever = MilvusColbertRetriever(milvus_client, collection_name)
        image_files = glob.glob(os.path.join(self.data_path, "*"))

        for i, image_file in enumerate(image_files):
            image_name = os.path.basename(image_file)
            embedding_file = os.path.join(self.embedding_path, image_name + ".npy")

            embeddings = np.load(embedding_file)
            embeddings = np.squeeze(embeddings, axis=0)

            data = {
                "doc_id": i,
                "colbert_vecs": embeddings,
                "filepath": image_file
            }

            milvus_colbert_retriever.insert(data)
    
    def add_text_data_to_milvus_db(self, collection_name: str, batch_size: int = 100, embedding_type: str = "bge_m3"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        json_paths = os.listdir(self.data_path)
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            bge_m3_embedding = BgeM3MilvusEmbedding()

            for json_path in json_paths:
                full_path = os.path.join(self.data_path, json_path)

                dict_content = json.load(open(full_path))
                content = processing_dict_data(dict_content) if isinstance(dict_content, dict) else str(dict_content)
                    
                chunked_docs = splitter.split_text(content)

                for i in range(0, len(chunked_docs), batch_size):
                    batch_data = []
                    batch_docs = chunked_docs[i:i + batch_size]
                    batch_embeddings = bge_m3_embedding.encode([doc for doc in batch_docs])
                    doc_id = json_path.split(".")[0]

                    sparse_vectors = []
                    for k in range(len(batch_docs)):
                        sparse_vector = batch_embeddings["sparse"][k]
                        sparse_vector = {c: v for c, v in zip(sparse_vector.col, sparse_vector.data)}
                        sparse_vectors.append(sparse_vector)

                    batch_data = {
                        "sparse_vector": sparse_vectors,
                        "dense_vector": batch_embeddings["dense"],
                        "text": batch_docs,
                        "doc_id": [doc_id] * len(batch_docs)
                    }
                            
                    milvus_bge_m3_retriever.insert(batch_data)

        elif embedding_type == "llm":
            time.sleep(120)
            llm_embedding = LLMEmbedding(model_name="models/gemini-embedding-exp-03-07")
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            
            for json_path in json_paths:
                full_path = os.path.join(self.data_path, json_path)

                dict_content = json.load(open(full_path))
                content = processing_dict_data(dict_content) if isinstance(dict_content, dict) else str(dict_content)
                    
                chunked_docs = splitter.split_text(content)

                for i in range(0, len(chunked_docs), batch_size):
                    batch_data = []
                    batch_docs = chunked_docs[i:i + batch_size]
                    batch_embeddings = llm_embedding.embed_batch_text([doc for doc in batch_docs])
                    doc_id = json_path.split(".")[0]

                    batch_data = {
                        "vector": batch_embeddings,
                        "text": batch_docs,
                        "doc_id": [doc_id] * len(batch_docs)
                    }
                            
                    milvus_llm_retriever.insert(batch_data)

    def add_table_data_to_milvus_db(self, collection_name: str, table_path: str, batch_size: int = 100, embedding_type: str = "bge_m3"):
        tb_description_files = os.listdir(self.data_path)
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            bge_m3_embedding = BgeM3MilvusEmbedding()

            for i in range(0, len(tb_description_files), batch_size):
                batch_data = []
                batch_files = tb_description_files[i:i + batch_size]

                tb_descriptions = []
                tb_text = []
                tb_ids = []

                for tb_description_file in batch_files:
                    with open(os.path.join(self.data_path, tb_description_file), "r") as f:
                        tb_descriptions.append(f.read())
                    
                    tb_description_name = tb_description_file.replace(".txt", "")
                    pdf_name = tb_description_name.split("_")[0]
                    table_id = "_".join(tb_description_name.split("_")[1:])
                    
                    with open(os.path.join(table_path, pdf_name + ".json"), "rb") as f:
                        data = json.load(f)

                        for tb_data in data:
                            if tb_data["table_id"] == table_id:
                                tb_text.append(str(tb_data))
                                tb_gt = [pdf_name.split(".")[0] + "_" + str(i) for i in range(tb_data["page_range"][0], tb_data["page_range"][1] + 1)] if len(tb_data["page_range"]) > 1 else [pdf_name.split(".")[0] + "_" + str(tb_data["page_range"][0])]
                                tb_ids.append(json.dumps(tb_gt))
                                break

                batch_embeddings = bge_m3_embedding.encode(tb_descriptions)

                sparse_vectors = []
                for k in range(len(tb_descriptions)):
                    sparse_vector = batch_embeddings["sparse"][k]
                    sparse_vector = {c: v for c, v in zip(sparse_vector.col, sparse_vector.data)}
                    sparse_vectors.append(sparse_vector)
                    
                batch_data = {
                    "sparse_vector": sparse_vectors,
                    "dense_vector": batch_embeddings["dense"],
                    "text": tb_text,
                    "doc_id": tb_ids
                }

                milvus_bge_m3_retriever.insert(batch_data)

        elif embedding_type == "llm":
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            llm_embedding = LLMEmbedding()

            for i in range(0, len(tb_description_files), batch_size):
                batch_data = []
                batch_files = tb_description_files[i:i + batch_size]

                tb_descriptions = []
                tb_text = []
                tb_ids = []

                for tb_description_file in batch_files:
                    with open(os.path.join(self.data_path, tb_description_file), "r") as f:
                        tb_descriptions.append(f.read())
                    
                    tb_description_name = tb_description_file.replace(".txt", "")
                    pdf_name = tb_description_name.split("_")[0]
                    table_id = "_".join(tb_description_name.split("_")[1:])
                    
                    with open(os.path.join(table_path, pdf_name + ".json"), "rb") as f:
                        data = json.load(f)

                        for tb_data in data:
                            if tb_data["table_id"] == table_id:
                                tb_text.append(str(tb_data))
                                tb_gt = [pdf_name.split(".")[0] + "_" + str(i) for i in range(tb_data["page_range"][0], tb_data["page_range"][1] + 1)] if len(tb_data["page_range"]) > 1 else [pdf_name.split(".")[0] + "_" + str(tb_data["page_range"][0])]
                                tb_ids.append(json.dumps(tb_gt))
                                break

                batch_embeddings = llm_embedding.embed_batch_text(tb_descriptions)
                    
                batch_data = {
                    "vector": batch_embeddings,
                    "text": tb_text,
                    "doc_id": tb_ids
                }

                milvus_llm_retriever.insert(batch_data)
            
    @staticmethod
    def group_by_prefix(arr):
        """
        Group array elements by their prefix (part before underscore).
        
        Args:
            arr (list): List of strings with format 'prefix_number'
            
        Returns:
            list: List of lists, where each inner list contains elements with same prefix
        """
        groups = {}
        for item in arr:
            prefix = item.split('_')[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(item)
        return list(groups.values())
