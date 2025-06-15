import pytest
from unittest.mock import Mock, patch
from backend.src.services.embedding_service.factories.implemenations.semantic_embedding_factory import SemanticEmbeddingServiceFactory
from backend.src.services.embedding_service.models.semantic_embedding.base import SemanticEmbeddingService
from backend.src.services.embedding_service.models.semantic_embedding.implementations.bge_m3_embedding import BGEM3EmbeddingService
from backend.src.services.embedding_service.models.semantic_embedding.implementations.colpali_embedding import ColPaliEmbeddingService
from backend.src.services.embedding_service.models.semantic_embedding.implementations.google_embedding import GoogleEmbeddingService
from backend.src.services.embedding_service.models.semantic_embedding.implementations.openai_embedding import OpenAIEmbeddingService

class TestSemanticEmbeddingServiceFactory:
    @pytest.fixture
    def factory(self):
        return SemanticEmbeddingServiceFactory()

    def test_get_available_services(self, factory):
        """Test that all expected services are available"""
        services = factory.get_available_services()
        expected_services = ["bge_m3", "colpali", "google", "openai"]
        assert sorted(services) == sorted(expected_services)

    @pytest.mark.parametrize("service_name,service_class", [
        ("bge_m3", BGEM3EmbeddingService),
        ("colpali", ColPaliEmbeddingService),
        ("google", GoogleEmbeddingService),
        ("openai", OpenAIEmbeddingService)
    ])
    def test_create_service_valid(self, factory, service_name, service_class):
        """Test creating valid services"""
        service = factory.create_service(service_name)
        assert isinstance(service, service_class)
        assert isinstance(service, SemanticEmbeddingService)

    def test_create_service_invalid(self, factory):
        """Test creating invalid service raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            factory.create_service("invalid_service")
        assert "Unknown service" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_register_service(self, factory):
        """Test registering a new service"""
        mock_service = Mock(spec=SemanticEmbeddingService)
        factory.register_service("test_service", type("TestService", (SemanticEmbeddingService,), {}))
        assert "test_service" in factory.get_available_services()

    def test_register_duplicate_service(self, factory):
        """Test registering duplicate service raises ValueError"""
        SemanticEmbeddingServiceFactory._services = {
            "bge_m3": BGEM3EmbeddingService,
            "colpali": ColPaliEmbeddingService,
            "google": GoogleEmbeddingService,
            "openai": OpenAIEmbeddingService
        }

        TestService = type("TestService", (SemanticEmbeddingService,), {})        
        factory.register_service("test_service", TestService)
        
        with pytest.raises(ValueError) as exc_info:
            factory.register_service("test_service", TestService)
        assert "already registered" in str(exc_info.value)