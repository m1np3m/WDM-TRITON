from typing import Dict, Type
from .base import LLMService
from .implementations.gemini import GeminiService

class LLMServiceFactory:
    """Factory for creating LLM service instances.
    
    Manages different LLM implementations through a registry pattern.
    """
    
    _services: Dict[str, Type[LLMService]] = {
        "gemini": GeminiService
    }
    
    @classmethod
    def register_service(cls, name: str, service_class: Type[LLMService]) -> None:
        """Register a new LLM service.
        
        Args:
            name: Service identifier
            service_class: LLM service class to register
            
        Raises:
            ValueError: If service name already exists
        """
        if name in cls._services:
            raise ValueError(f"Service '{name}' is already registered")
        cls._services[name] = service_class
    
    @classmethod
    def create_service(cls, name: str, **kwargs) -> LLMService:
        """Create an LLM service instance.
        
        Args:
            name: Name of service to create
            **kwargs: Arguments for service constructor
            
        Returns:
            LLMService instance
            
        Raises:
            ValueError: If service not found
        """
        if name not in cls._services:
            available = ", ".join(cls.get_available_services())
            raise ValueError(f"Unknown service: '{name}'. Available: {available}")
        
        return cls._services[name](**kwargs)
    
    @classmethod
    def get_available_services(cls) -> list[str]:
        """Get list of registered service names."""
        return list(cls._services.keys())
