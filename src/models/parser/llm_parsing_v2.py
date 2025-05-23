from src.services.llm import LLMService
import json
import os

class LLMParser:
    def __init__(self, data_path: str):
        self.llm_service = LLMService()
        self.data_path = data_path

        self.providers = [{"name": "gemini-file", "model": "gemini-2.5-flash-preview-05-20", "temperature": 0.9, "retry": 3}]

        self.SYSTEM_PARSING_PROMPT = """
        ### ROLE
            - You are a document parsing tool.

        ### INSTRUCTIONS
            - You will receive a document.
            - Extract all tables from this document, including those that span across multiple pages.
                + Preserve the original row and column structure.
                + If a table continues on the next page(s), merge it properly with the corresponding table segment.
                + If table headers are repeated across pages, remove the duplicated headers in the final output.

        ### OUTPUT
            - Return the results in JSON format structured as follows:
            [
            {
                "table_id": "Table_1",
                "page_range": [2, 3],
                "columns": ["Column 1", "Column 2", "Column 3"],
                "rows": [
                ["Row 1 - Col 1", "Row 1 - Col 2", "Row 1 - Col 3"],
                ["Row 2 - Col 1", "Row 2 - Col 2", "Row 2 - Col 3"]
                ]
            },
            ...
            ]

        ### REQUIREMENTS
            - Each table should include a unique table_id and a page_range indicating which pages the table spans.
            - If column headers are not clearly defined, try to infer them from the first row of the table.
            - Make sure no tables are skipped, including those without clear borders.
        """

    def parse(self, output_path: str) -> str:
        pdf_paths = os.listdir(self.data_path)
        for pdf_path in pdf_paths:
            for attempt in range(3):
                parsing_results = self.llm_service.complete(
                                system_prompt=self.SYSTEM_PARSING_PROMPT,
                                user_prompt="Please extract all tables from the document",
                                file_paths=[os.path.join(self.data_path, pdf_path)],
                                json_output=True,
                                providers=self.providers,
                                )
                
                json_result = json.loads(json.dumps(parsing_results))

                if not json_result:
                    print(f"Failed to parse document {pdf_path} after {attempt + 1} attempts")
                    continue
                break

            # write json_result to a file
            os.makedirs(output_path, exist_ok=True)
            pdf_name = os.path.basename(pdf_path)

            with open(f"{os.path.join(output_path, pdf_name)}.json", "w") as f:
                json.dump(json_result, f, indent=4)