import glob
import os
import numpy as np
from scipy.sparse import csr_matrix
from src.datasets.dataset import Dataset
from src.db import chroma_client
from src.db.milvus import MilvusColbertRetriever, MilvusBgeM3Retriever
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.models.embedding.bge_m3 import BgeM3MilvusEmbedding

from src.services.processing.text_processing import merge_json_data


milvus_client = MilvusClient(uri="milvus_db/milvus.db")

class M3DocVQA(Dataset):
    def __init__(self, data_path: str, embedding_path: str=None):
        super().__init__(data_path, embedding_path)

    def load_data(self):
        self.data = []

    def add_copali_data_to_chroma_db(self, collection_name: str, batch_size: int = 100):
        image_files = glob.glob(os.path.join(self.data_path, "*"))
        
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

            chroma_client.add_image_embeddings(collection_name, batch_data)

    def get_data(self):
        return self.data

    def add_copali_data_to_milvus_db(self, collection_name: str):
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
    
    def add_text_data_to_milvus_db(self, collection_name: str, batch_size: int = 100):
        milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
        bge_m3_embedding = BgeM3MilvusEmbedding()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        json_paths = os.listdir(self.data_path)

        pdf_paths = self.group_by_prefix(json_paths)
        for pdf_path in pdf_paths:
            full_paths = [os.path.join(self.data_path, path) for path in pdf_path]

            full_text = merge_json_data(full_paths)
            chunked_docs = splitter.split_text(full_text)

            for i in range(0, len(chunked_docs), batch_size):
                batch_data = []
                batch_docs = chunked_docs[i:i + batch_size]
                batch_embeddings = bge_m3_embedding.encode([doc for doc in batch_docs])
                doc_id = pdf_path[0].split("_")[0]

                sparse_vectors = []
                for i in range(len(batch_docs)):
                    sparse_vector = batch_embeddings["sparse"][i]
                    sparse_vector = {c: v for c, v in zip(sparse_vector.col, sparse_vector.data)}
                    sparse_vectors.append(sparse_vector)

                batch_data = {
                    "sparse_vector": sparse_vectors,
                    "dense_vector": batch_embeddings["dense"],
                    "text": batch_docs,
                    "doc_id": [doc_id] * len(batch_docs)
                }
                
                milvus_bge_m3_retriever.insert(batch_data)
                    
    def add_table_data_to_milvus_db(self, collection_name: str):
        pass

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
