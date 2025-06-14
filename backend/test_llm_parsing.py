from src.services.parsing_service.llm_parsing import LLMParsingService

llm_parsing_service = LLMParsingService(
    llm_service_name="gemini",
    model="gemini-1.5-flash"
)

llm_parsing_service.parse_file(
    pdf_filename="0aed309e29e45111f67fb85aea1fcb5e.pdf",
    data_dir="D:/WDM-TRITON/backend/data/pdfs/",
    output_dir="D:/WDM-TRITON/backend/data/test"
)