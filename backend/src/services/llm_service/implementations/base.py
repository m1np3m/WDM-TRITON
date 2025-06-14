from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM services."""
    def __init__(self):
        self._model = None
        self._api_key = None
        self._client = None
        self._connection = None
        self._is_connected = False
    
    @abstractmethod
    def _connect(self) -> None:
        """Set up LLM connection."""
        pass
    
    @abstractmethod
    def _connect(self) -> None:
        """Set up LLM connection."""
        pass
    
    def ensure_connection(self) -> None:
        """Ensure connection, reconnect if not connected."""
        if not self.is_connected():
            self._connect()
            self._is_connected = True
    
    @abstractmethod
    def generate(self) -> None:
        pass
    
    @abstractmethod
    def generate_with_image(self) -> None:
        pass
    
    @abstractmethod
    def generate_with_file(self) -> None:
        pass