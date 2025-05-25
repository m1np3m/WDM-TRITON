import chromadb
import numpy as np
import os
import dotenv

dotenv.load_dotenv()
client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))

def create_collection(name: str):
    return client.create_collection(name)

def get_collection(name: str):
    return client.get_or_create_collection(name)

def add_image_embeddings(collection_name: str, batch_data: list):
    ids = [data["id"] for data in batch_data]
    embeddings = [data["embedding"] for data in batch_data]
    image_names = [data["image_name"] for data in batch_data]

    return get_collection(collection_name).add(
            ids=ids,
            embeddings=embeddings,
            metadatas=[{"image_name": image_name} for image_name in image_names]
        )

def query_image_embeddings(collection_name: str, query_embedding: np.ndarray, n_results: int = 5):
    return get_collection(collection_name).query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

def delete_collection(collection_name: str):
    return client.delete_collection(collection_name)
