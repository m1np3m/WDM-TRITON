from typing import Dict, Type, List
from .base import BaseDatabase
from .implementations.milvus import MilvusDB

class DatabaseFactory:
    """
    Factory class to manage registration and instantiation of vector database implementations.
    """
    
    _databases: Dict[str, Type[BaseDatabase]] = {
        "milvus": MilvusDB
    }

    @classmethod
    def register_database(cls, name: str, database_class: Type[BaseDatabase]) -> None:
        """
        Register a new database implementation under a given name.

        Args:
            name (str): Unique name for the database.
            database_class (Type[BaseDatabase]): Class implementing the BaseDatabase interface.

        Raises:
            ValueError: If the name has already been registered.
        """
        if name in cls._databases:
            raise ValueError(f"Database '{name}' is already registered.")
        cls._databases[name] = database_class

    @classmethod
    def create_service(cls, name: str, **kwargs) -> BaseDatabase:
        """
        Instantiate a registered database implementation.

        Args:
            name (str): Name of the registered database.
            **kwargs: Arguments to be passed to the database constructor.

        Returns:
            BaseDatabase: An instance of the requested database.

        Raises:
            ValueError: If the database name is not recognized.
        """
        if name not in cls._databases:
            available = ", ".join(cls.get_available_databases())
            raise ValueError(f"Unknown service: '{name}'. Available: {available}")
        
        return cls._databases[name](**kwargs)

    @classmethod
    def get_available_databases(cls) -> List[str]:
        """
        Get a list of all registered database names.

        Returns:
            List[str]: List of available database names.
        """
        return list(cls._databases.keys())
