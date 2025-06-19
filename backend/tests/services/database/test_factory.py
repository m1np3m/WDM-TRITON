import pytest
from backend.src.database.factory import DatabaseFactory
from backend.src.database.base import BaseDatabase
from backend.src.database.implementations.milvus import MilvusDB

class DummyDB(BaseDatabase):
    def _connect(self): pass
    def get_collection(self, collection_name): pass
    def create_collection(self, collection_name, schema): pass
    def index_document(self, collection_name, document, doc_id): pass
    def search(self, collection_name, query): pass
    def delete_collection(self, collection_name): pass
    def delete_document(self, collection_name, doc_id): pass
    def update_document(self, collection_name, document, doc_id): pass
    def close(self): pass
    def ping(self): return True

@pytest.fixture(autouse=True)
def reset_factory():
    # Reset lại _databases về trạng thái ban đầu trước mỗi test
    DatabaseFactory._databases = {"milvus": MilvusDB}
    yield
    DatabaseFactory._databases = {"milvus": MilvusDB}

def test_get_available_databases():
    available = DatabaseFactory.get_available_databases()
    assert "milvus" in available

def test_create_service_valid():
    db = DatabaseFactory.create_service("milvus")
    assert isinstance(db, MilvusDB)
    assert isinstance(db, BaseDatabase)

def test_create_service_invalid():
    with pytest.raises(ValueError) as exc_info:
        DatabaseFactory.create_service("not_exist")
    assert "Unknown service" in str(exc_info.value)
    assert "Available" in str(exc_info.value)

def test_register_database():
    name = "dummy"
    DatabaseFactory.register_database(name, DummyDB)
    assert name in DatabaseFactory.get_available_databases()
    db = DatabaseFactory.create_service(name)
    assert isinstance(db, DummyDB)

def test_register_duplicate_database():
    name = "dummy_duplicate"
    DatabaseFactory.register_database(name, DummyDB)
    with pytest.raises(ValueError) as exc_info:
        DatabaseFactory.register_database(name, DummyDB)
    assert "already registered" in str(exc_info.value)