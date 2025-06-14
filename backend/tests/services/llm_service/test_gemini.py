import os
import pytest
from unittest.mock import Mock, patch
from google.genai.types import GenerateContentResponse

from backend.src.services.llm_service.implementations.gemini import GeminiService

@pytest.fixture
def mock_genai_client():
    with patch('google.genai.Client') as mock_client:
        yield mock_client

@pytest.fixture
def gemini_service(mock_genai_client):
    # Set test API key
    os.environ['GEMINI_API_KEY'] = 'test-api-key'
    service = GeminiService()
    return service

def test_init_with_env_api_key():
    os.environ['GEMINI_API_KEY'] = 'test-api-key'
    service = GeminiService()
    assert service._api_key == 'test-api-key'

def test_init_with_direct_api_key():
    service = GeminiService(api_key='direct-api-key')
    assert service._api_key == 'direct-api-key'

def test_init_without_api_key():
    if 'GEMINI_API_KEY' in os.environ:
        del os.environ['GEMINI_API_KEY']
    
    with pytest.raises(ValueError) as exc_info:
        GeminiService()
    assert "API key not provided" in str(exc_info.value)

def test_generate_text_only(gemini_service, mock_genai_client):
    # Mock response
    mock_response = Mock(spec=GenerateContentResponse)
    mock_response.text = '{"result": "test response"}'
    
    # Setup mock
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.return_value = mock_response
    
    # Test
    result = gemini_service.generate(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt"
    )
    
    assert result["is_success"] is True
    assert result["response"] == {"result": "test response"}
    assert result["error"] is None

def test_generate_with_files(gemini_service, mock_genai_client, tmp_path):
    # Tạo test files
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Mock response
    mock_response = Mock(spec=GenerateContentResponse)
    mock_response.text = '{"result": "test response with file"}'
    
    # Setup mocks
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.chats.create.return_value.send_message.return_value = mock_response
    
    # Mock file upload
    mock_file = Mock()
    mock_file.uri = "test-uri"
    mock_client_instance.files.upload.return_value = mock_file
    
    # Test
    result = gemini_service.generate(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        file_paths=[str(test_file)]
    )
    
    assert result["is_success"] is True
    assert result["response"] == {"result": "test response with file"}
    assert result["error"] is None

def test_generate_error_handling(gemini_service, mock_genai_client):
    # Setup mock để raise exception
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception("Test error")
    
    # Test
    result = gemini_service.generate(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        retry=0  # Disable retry for testing
    )
    
    assert result["is_success"] is False
    assert result["response"] is None
    assert "Test error" in result["error"]

def test_invalid_json_response(gemini_service, mock_genai_client):
    # Mock response với JSON không hợp lệ
    mock_response = Mock(spec=GenerateContentResponse)
    mock_response.text = "invalid json"
    
    # Setup mock
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.return_value = mock_response
    
    # Test
    result = gemini_service.generate(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        retry=0  # Disable retry for testing
    )
    
    assert result["is_success"] is False
    assert result["response"] is None
    assert "JSON decode error" in result["error"] 