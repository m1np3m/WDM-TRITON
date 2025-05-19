class Dataset:
    def __init__(self, data_path: str, embedding_path: str):
        self.data_path = data_path
        self.embedding_path = embedding_path

    def load_data(self):
        pass

    def save_data(self):
        pass

    def add_data_to_db(self, collection_name: str, batch_size: int = 100):
        pass

    def get_data(self):
        pass
