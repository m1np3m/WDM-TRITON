from typing import Dict, Type, List

from ..models.base import EmbeddingService

class EmbeddingServiceFactory:
    """
    Factory to register and instantiate embedding services.
    """
    _services: Dict[str, Type[EmbeddingService]] = {}

    @classmethod
    def register_service(cls, name: str, service_class: Type[EmbeddingService]) -> None:
        """
        Register a new embedding service class.

        Args:
            name (str): Identifier for the service.
            service_class (Type[EmbeddingService]): Service class to register.

        Raises:
            ValueError: If the service name is already registered.
        """
        if name in cls._services:
            raise ValueError(f"Service '{name}' is already registered")
        cls._services[name] = service_class

    @classmethod
    def create_service(cls, name: str, **kwargs) -> EmbeddingService:
        """
        Create an instance of a registered embedding service.

        Args:
            name (str): Identifier of the service.
            **kwargs: Arguments passed to the service constructor.

        Returns:
            EmbeddingService: An instance of the requested service.

        Raises:
            ValueError: If the service is not registered.
        """
        if name not in cls._services:
            available = ", ".join(cls.get_available_services())
            raise ValueError(f"Unknown service: '{name}'. Available: {available}")
        return cls._services[name](**kwargs)

    @classmethod
    def get_available_services(cls) -> List[str]:
        """
        List all registered embedding service names.

        Returns:
            List[str]: Names of all registered services.
        """
        return list(cls._services.keys())
