import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.src.services.parsing_service.llm_parsing import LLMParsingService

@pytest.fixture
def mock_llm_service():
    with patch('backend.src.services.llm_service.factory.LLMServiceFactory.create_service') as mock:
        mock_service = Mock()
        mock.return_value = mock_service
        yield mock_service

@pytest.fixture
def parsing_service(mock_llm_service):
    return LLMParsingService()

@pytest.fixture
def temp_dirs(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    return data_dir, output_dir

def test_init_default_values(parsing_service):
    assert parsing_service.system_prompt == LLMParsingService.DEFAULT_SYSTEM_PROMPT
    assert parsing_service.llm_service is not None

def test_init_custom_values():
    custom_prompt = "Custom system prompt"
    service = LLMParsingService(system_prompt=custom_prompt)
    assert service.system_prompt == custom_prompt

def test_get_unprocessed_pdfs_empty_dir(temp_dirs):
    data_dir, output_dir = temp_dirs
    service = LLMParsingService()
    unprocessed = service._get_unprocessed_pdfs(data_dir, output_dir)
    assert len(unprocessed) == 0

def test_get_unprocessed_pdfs_with_files(temp_dirs):
    data_dir, output_dir = temp_dirs
    
    # Tạo file PDF test
    (data_dir / "test1.pdf").touch()
    (data_dir / "test2.pdf").touch()
    
    service = LLMParsingService()
    unprocessed = service._get_unprocessed_pdfs(data_dir, output_dir)
    assert len(unprocessed) == 2
    assert "test1.pdf" in unprocessed
    assert "test2.pdf" in unprocessed

def test_get_unprocessed_pdfs_with_existing_output(temp_dirs):
    data_dir, output_dir = temp_dirs
    
    # Tạo file PDF và output tương ứng
    (data_dir / "test1.pdf").touch()
    (data_dir / "test2.pdf").touch()
    (output_dir / "test1.pdf.json").touch()
    
    service = LLMParsingService()
    unprocessed = service._get_unprocessed_pdfs(data_dir, output_dir)
    assert len(unprocessed) == 1
    assert "test2.pdf" in unprocessed

def test_parse_file_success(parsing_service, temp_dirs, mock_llm_service):
    data_dir, output_dir = temp_dirs
    pdf_path = data_dir / "test.pdf"
    pdf_path.touch()
    
    # Mock LLM response
    mock_response = {
        "is_success": True,
        "response": [
            {
                "table_id": "table_1",
                "page_range": [1, 2],
                "columns": ["col1", "col2"],
                "rows": [["data1", "data2"]]
            }
        ]
    }
    mock_llm_service.generate.return_value = mock_response
    
    # Test parse file
    parsing_service.parse_file(
        pdf_filename="test.pdf",
        data_dir=data_dir,
        output_dir=output_dir,
        user_prompt="Test prompt"
    )
    
    output_file = output_dir / "test.pdf.json"
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == mock_response["response"]

def test_parse_file_retry_on_failure(parsing_service, temp_dirs, mock_llm_service):
    data_dir, output_dir = temp_dirs
    pdf_path = data_dir / "test.pdf"
    pdf_path.touch()
    
    mock_llm_service.generate.side_effect = [
        Exception("First attempt failed"),
        Exception("Second attempt failed"),
        {
            "is_success": True,
            "response": [{"table_id": "table_1", "rows": []}]
        }
    ]
    
    parsing_service.parse_file(
        pdf_filename="test.pdf",
        data_dir=data_dir,
        output_dir=output_dir,
        user_prompt="Test prompt"
    )
    
    assert mock_llm_service.generate.call_count == 3
    assert (output_dir / "test.pdf.json").exists()

def test_process_parsing_results_valid_json(parsing_service):
    test_data = [{"table_id": "table_1", "rows": []}]
    
    result = parsing_service._process_parsing_results(test_data)
    assert result == test_data

    json_str = json.dumps(test_data)
    result = parsing_service._process_parsing_results(json_str)
    assert result == test_data

def test_process_parsing_results_invalid_json(parsing_service):
    invalid_data = "invalid json"
    result = parsing_service._process_parsing_results(invalid_data)
    assert result is None

def test_save_results(parsing_service, temp_dirs):
    _, output_dir = temp_dirs
    test_data = [{"table_id": "table_1", "rows": []}]
    
    parsing_service._save_results(
        pdf_filename="test.pdf",
        json_result=test_data,
        output_dir=output_dir
    )
    
    output_file = output_dir / "test.pdf.json"
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == test_data 