import glob
import os
import numpy as np
from src.datasets.dataset import Dataset
from src.db import chroma_client

class M3DocVQA(Dataset):
    def __init__(self, image_path: str, embedding_path: str):
        super().__init__(image_path, embedding_path)

    def load_data(self):
        self.data = []

    def add_data_to_db(self, collection_name: str, batch_size: int = 100):
        image_files = glob.glob(os.path.join(self.image_path, "*"))
        
        # Process files in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_data = []
            
            img_num = 0
            for image_file in batch_files:
                image_name = os.path.basename(image_file)
                embedding_file = os.path.join(self.embedding_path, image_name)

                embedding = np.load(embedding_file)
                embedding = np.squeeze(embedding, axis=0)
                aggregated_embedding = np.mean(embedding, axis=0)
                aggregated_embedding = aggregated_embedding.tolist()

                batch_data.append({"id": f"batch_{i}_{img_num}", "image_name": image_name, "embedding": aggregated_embedding})
                img_num += 1

            chroma_client.add_image_embeddings(collection_name, batch_data)

    def get_data(self):
        return self.data
