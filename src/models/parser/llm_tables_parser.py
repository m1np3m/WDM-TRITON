from src.services.llm import LLMService
import json
import os
from google import genai
import glob
from google.genai.types import GenerateContentConfig

@staticmethod
def group_by_prefix(arr):
    """
    Group array elements by their prefix (part before underscore).
        
    Args:
        arr (list): List of strings with format 'prefix_number'
            
    Returns:
        list: List of lists, where each inner list contains elements with same prefix
    """
    groups = {}
    for item in arr:
        prefix = item.split('_')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(item)
    return list(groups.values())

class LLMTableParser:
    def __init__(self, data_path: str):
        self.llm_service = LLMService()
        self.data_path = data_path

        self.providers = [{"name": "gemini-image", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.9, "retry": 3}]
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.5-flash-preview-04-17"

        self.SYSTEM_PARSING_PROMPT = """
        ### ROLE
            - You are a document parsing tool.

        ### INSTRUCTIONS
            - You will receive image documents that can contain many different objects such as: text, images, tables, charts, ...
            - Extract all tables from the image and convert them into JSON format.
            - Preserve the order of the tables exactly as they appear in the original image, from top to bottom and left to right.
            - If the table has column headers, include them as a separate field for clarity.
            - Ensure the JSON structure is clean, well-formatted, and easy to use for further data processing.
            - If the table does not have a header, leave the header blank.

        ### OUTPUT
            - Return the results in JSON format structured as follows:
            [
            {
                "table_id": "Table_1",
                "columns": ["Column 1", "Column 2", "Column 3"],
                "rows": [
                ["Row 1 - Col 1", "Row 1 - Col 2", "Row 1 - Col 3"],
                ["Row 2 - Col 1", "Row 2 - Col 2", "Row 2 - Col 3"]
                ]
            },
            ...
            ]
        """

        self.SYSTEM_MERGING_PROMPT = """
        ### ROLE
            - You are a document analysis tool.

        ### INSTRUCTIONS
            - You will receive a list of tables extracted from multiple pages of a document, each in this JSON format:

            [
            {
                "table_id": "Table_1",
                "page": 1,
                "columns": ["Column 1", "Column 2", "Column 3"],
                "rows": [
                ["Row 1 - Col 1", "Row 1 - Col 2", "Row 1 - Col 3"],
                ...
                ]
            },
            ...
            ]

            - Your task is to:
                1. Analyze the content of each table and identify tables that represent continuations or parts of the same logical table but appear on different pages, even if their `table_id` values differ.
                2. Merge such related tables into one combined table by:
                    - Using the columns from the first occurrence (assume columns are consistent or very similar).
                    - Combining all rows from all pages in the correct order.
                    - Creating a new `"page_range"` field showing the smallest and largest page numbers covered.
                    - Assigning a unified `"table_id"` by choosing the earliest encountered table's `table_id`.

                3. Tables that are unique or unrelated should remain as separate entries.

        ### OUTPUT
            - Return the merged tables as a JSON array with the structure:
            [
            {
                "table_id": "Table_1",
                "columns": [...],
                "page_range": [1, 2],
                "rows": [...]
            },
            ...
            ]
            - Remember to create "page_range" field showing the smallest and largest page numbers covered.
        """

    def parse(self, output_path: str) -> str:
        image_paths = glob.glob(os.path.join(self.data_path, "*.png"))
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            if not os.path.exists(f"{os.path.join(output_path, image_name)}.json"):
                for attempt in range(3):
                    # parsing_results = self.llm_service.complete(
                    #                 system_prompt=self.SYSTEM_PARSING_PROMPT,
                    #                 user_prompt="Please parse the content of this document image in detail",
                    #                 image_paths=[image_path],
                    #                 json_output=True,
                    #                 providers=self.providers,
                    #                 )
                    img_file = self.client.files.upload(file=image_path)
                    chat_session = self.client.chats.create(model=self.model, config=GenerateContentConfig(system_instruction=self.SYSTEM_PARSING_PROMPT, response_mime_type="application/json"))
                    chat_session.send_message(img_file)
                    response = chat_session.send_message("Please extract all tables in detail")
                    parsing_results = response.text

                    try:
                        json_result = json.loads(parsing_results)
                    except json.JSONDecodeError as jde:
                        print(f"JSON decode error: {jde}")
                        if attempt == 2:
                            json_result = None
                        continue

                    if not json_result:
                        print(f"Failed to parse image {image_path} after {attempt + 1} attempts")
                        if attempt == 2:
                            json_result = None
                        continue

                    break

                # write json_result to a file
                os.makedirs(output_path, exist_ok=True)

                with open(f"{os.path.join(output_path, image_name)}.json", "w") as f:
                    json.dump(json_result, f, indent=4)
    
    def merge(self, input_path: str, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        json_names = os.listdir(input_path)
        json_names = group_by_prefix(json_names)

        for _name in json_names:
            all_tables = []
            for json_name in _name:
                with open(os.path.join(input_path, json_name), "r") as f:
                    json_result = json.load(f)

                if json_result:
                    for table in json_result:
                        if len(table["columns"]) > 0:
                            table["rows"].insert(0, table["columns"])
                        table["page"] = int(json_name.split("_")[1].split(".")[0])
                    all_tables.extend(json_result)
            
            if all_tables:
                for attempt in range(5):
                    merge_results = self.client.models.generate_content(
                        contents=[f"Please merge the following tables: {all_tables}"],
                        model=self.model,
                            config=GenerateContentConfig(system_instruction=self.SYSTEM_MERGING_PROMPT, response_mime_type="application/json")
                        )
                    merge_results = merge_results.text

                    try:    
                        merge_results = json.loads(merge_results)
                        # retry if merge_results not have "page_range" field
                        for table in merge_results:
                            if "page_range" not in table:
                                print(f"Page range not found in table. Retry {attempt + 1}")
                                continue
                    except json.JSONDecodeError as jde:
                        print(f"JSON decode error: {jde}")
                        if attempt == 4:
                            print(f"Failed to merge tables {_name} after {attempt + 1} attempts")
                            merge_results = None
                        continue
                    break
            
            if isinstance(merge_results, list):
                for i, table in enumerate(merge_results):
                    table["table_id"] = f"Table_{i+1}"

            output_name = f"{_name[0].split('_')[0]}.json"
            with open(os.path.join(output_path, output_name), "w") as f:
                json.dump(merge_results, f, indent=4)
                    
class LLMTableParserV2:
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
        pdf_paths = [pdf_path for pdf_path in pdf_paths if f"{os.path.basename(pdf_path)}.json" not in os.listdir(output_path)]
        print(f"Found {len(pdf_paths)} pdfs to parse")

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