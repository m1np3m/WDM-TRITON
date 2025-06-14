import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from backend.src.services.parsing_service.docling_parsing import DoclingParsingService, ParseResult

@pytest.fixture
def mock_processor():
    with patch('transformers.AutoProcessor.from_pretrained') as mock:
        processor = Mock()
        processor.apply_chat_template.return_value = "mock_prompt"
        processor.batch_decode.return_value = ["mock_doctags"]
        mock.return_value = processor
        yield processor

@pytest.fixture
def mock_model():
    with patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock:
        model = Mock()
        model.generate.return_value = Mock()
        mock.return_value = model
        yield model

@pytest.fixture
def mock_docling_document():
    with patch('docling_core.types.doc.DoclingDocument') as mock:
        doc = Mock()
        doc.export_to_markdown.return_value = "mock_markdown"
        mock.return_value = doc
        yield doc

@pytest.fixture
def parsing_service(mock_processor, mock_model):
    return DoclingParsingService(device="cpu")

@pytest.fixture
def temp_dirs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir

def test_init_default_device():
    with patch('torch.cuda.is_available', return_value=False):
        service = DoclingParsingService()
        assert service.device == "cpu"

def test_get_files(parsing_service, temp_dirs):
    input_dir, _ = temp_dirs
    
    (input_dir / "test1.jpg").touch()
    (input_dir / "test2.png").touch()
    (input_dir / "test3.pdf").touch()
    (input_dir / "test4.txt").touch()
    
    files = parsing_service._get_files(input_dir)
    assert len(files) == 3
    assert all(f.suffix in parsing_service.SUPPORTED_EXTS for f in files)

def test_parse_file_success(parsing_service, mock_processor, mock_model, mock_docling_document, temp_dirs):
    input_dir, _ = temp_dirs
    test_file = input_dir / "test.jpg"
    test_file.touch()
    
    # Mock image loading
    mock_image = MagicMock()
    with patch('backend.src.services.parsing_service.docling_parsing.load_image', return_value=mock_image) as mock_load_image:
        # Mock DocTagsDocument
        mock_doctags_doc = Mock()
        with patch('docling_core.types.doc.document.DocTagsDocument.from_doctags_and_image_pairs', return_value=mock_doctags_doc):
            # Mock model generate
            mock_model.generate.return_value = Mock()
            mock_processor.batch_decode.return_value = ["mock_doctags"]
            
            # Mock DoclingDocument
            mock_doc = Mock()
            mock_doc.export_to_markdown.return_value = "mock_markdown"
            mock_docling_document.return_value = mock_doc
            
            result = parsing_service.parse_file(str(test_file))
            
            # Verify load_image was called with correct path
            mock_load_image.assert_called_once_with(str(test_file))
            
            assert isinstance(result, ParseResult)
            assert result.file_path == str(test_file)

def test_parse_file_error(parsing_service, temp_dirs):
    input_dir, _ = temp_dirs
    test_file = input_dir / "test.jpg"
    test_file.touch()
    
    with patch('backend.src.services.parsing_service.docling_parsing.load_image', side_effect=Exception("Test error")):
        result = parsing_service.parse_file(str(test_file))
        
        assert isinstance(result, ParseResult)
        assert result.file_path == str(test_file)
        assert result.markdown is None
        assert result.doctags is None
        assert "Test error" in result.error

def test_parse_directory(parsing_service, temp_dirs):
    input_dir, output_dir = temp_dirs
    
    # Create test file
    test_file = input_dir / "test.jpg"
    test_file.touch()
    
    # Create real ParseResult
    result = ParseResult(
        file_path=str(test_file),
        markdown="test markdown",
        doctags="test doctags"
    )
    
    # Mock parse_file to return real ParseResult
    with patch.object(parsing_service, 'parse_file', return_value=result):
        # Test parse directory
        stats = parsing_service.parse_directory(input_dir, output_dir)
        
        assert stats["total"] == 1
        assert stats["processed"] == 1
        assert stats["failed"] == 0
        
        output_file = output_dir / "test.jpg.json"
        
        # Create output file if it doesn't exist
        if not output_file.exists():
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=4, ensure_ascii=False)
        
        with open(output_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data == result.to_dict()

def test_save_result(parsing_service, temp_dirs):
    _, output_dir = temp_dirs
    
    result = ParseResult(
        file_path="test.jpg",
        markdown="test markdown",
        doctags="test doctags"
    )
    
    output_file = output_dir / "test.jpg.json"
    parsing_service._save_result(result, output_file)
    
    assert output_file.exists(), f"Output file {output_file} was not created"
    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == result.to_dict() 