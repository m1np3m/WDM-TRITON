class Dataset:
    def __init__(self, image_path: str, embedding_path: str):
        self.image_path = image_path
        self.embedding_path = embedding_path

    def load_data(self):
        pass

    def save_data(self):
        pass

    def add_data_to_db(self, collection_name: str, batch_size: int = 100):
        pass

    def get_data(self):
        pass
