import csv
from datetime import datetime
import glob
import json
import os
from typing import Any, Dict, List
from src.services.llm import LLMService
from loguru import logger
from google import genai
from google.genai.types import GenerateContentConfig

llm_service = LLMService()

providers = [{"name": "gemini", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.0, "retry": 3}]

MERGE_SYSTEM_PROMPT = """
    ### ROLE
	    - You are a table processing tool.

    ### INSTRUCTIONS
        - You will get a dictionary containing information about the tables with the order of appearance of the tables corresponding to the order of appearance in the dictionary:
        - The dictionary will include:
            + key: Table name.
            + value: The table will be represented in csv format including the following information:
                1. The first few rows of data in the table.
                2. The last few rows of data in the table.
		
        - Your task is to find all the tables that can be merged together.
        - You must give a reason why you are merging these tables.

    ### OUTPUT
        - Your OUTPUT must follow JSON format, in which:
            + key will include the names of the tables that can be merged together, separated by "-".
            + value will be the reason why you are merging these tables.

        - For example: 
        { 
        "0a89f965fee003cf40c716e27bb0c1a9_1.table_7-0a89f965fee003cf40c716e27bb0c1a9_10.table_1-0a89f965fee003cf40c716e27bb0c1a9_3.table_1": "" 
        }
        - If no tables can be merged, return {}
        
    ### CHAIN OF THOUGHT
        1. You need to carefully check all the table's appearance positions to determine which part comes first and which part comes after if you merge them.
        2. You need to carefully consider the content of each table.
        3. Check the Header (if any) of each table to serve as a basis for merging the table.
        4. Based on the acquired knowledge, perform the assigned request
"""

DESCRIPTION_SYSTEM_PROMPT = """
    ### ROLE
	    - You are a table descriptor.

    ### INSTRUCTION
        - You will receive a table in CSV format. Your task is to write a short sentence describing each row of data in this table.
        - You must ensure that you provide a complete description for all rows in the table, without leaving anything out.
        - You should not return empty results, try to describe the data received.

    ### OUTPUT
        - Your OUTPUT must follow the JSON format, in which:
            + key: number of rows of data in table.
            + value: Describes the data in that row.
        
        For example:
        {
            "0": "...",
            "1": "..."
        }
"""

def extract_table_data(data_paths):
    # Nhận vào toàn bộ paths json parsing từ 1 tài liệu -> lấy ra hết table -> gửi lên LLM -> lấy kết quả rồi gộp table lại.

    small_tables = {}
    tables = {}
    for data_path in data_paths:
        with open(data_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        for key, value in data.items():
            if "table" in key:
                tb_key = os.path.basename(data_path).split(".")[0] + "." + key
                small_tables[tb_key] = get_small_table(value)
                tables[tb_key] = value

    if len(tables) > 1:
        logger.info(f"Merging tables")
        ### Get merge info from LLM
        merge_info = llm_service.complete(
            system_prompt=MERGE_SYSTEM_PROMPT,
            user_prompt=f"Please get information about tables that can be merged from tables: {json.dumps(small_tables)}",
            json_output=True,
            providers=providers,
        )
        
        if merge_info:
            try:
                ### Merge tables
                for key, value in merge_info.items():
                    tb_keys = key.split("-")
                    merge_value = ""
                    for tb_key in tb_keys:
                        merge_value += f"\n{tables[tb_key]}"
                        del tables[tb_key]
                    tables[key] = merge_value
            except Exception as e:
                logger.error(f"Error merging tables: {e}")

    # logger.info(f"Getting description for each table")
    # ### Get description for each table
    # for key, value in tables.items():
    #     tb_description = get_table_description(value)
    #     if tb_description:
    #         tables[key] = tb_description
    #     else:
    #         tables[key] = {"0": value}

    return tables

def get_table_description(table_data: str) -> str:
    description = llm_service.complete(
        system_prompt=DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=table_data,
        json_output=True,
        providers=providers,
    )

    return description

def get_small_table(table_data: str) -> str:
    table_rows = table_data.split("\n")
    if len(table_rows) == 0:
        table_rows = table_data
    
    t_reader = csv.reader(table_rows)

    # Nếu số dòng < 6 thì lấy hết
    t_reader = list(t_reader)
    if len(t_reader) < 6:
        result = t_reader
    else:
        result = t_reader[:3] + t_reader[-3:]
    
    # Chuyển đổi list thành chuỗi CSV
    output = []
    for row in result:
        output.append(','.join(row))
    return '\n'.join(output)

def table_to_text(data_path: str) -> str:
    with open(data_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return ""

class TableRowDescription:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.llm_service = LLMService()
        self.providers = [{"name": "gemini", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.0, "retry": 3}]
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.5-flash-preview-04-17"

        self.DESCRIPTION_SYSTEM_PROMPT = """
            ### ROLE
                - You are a document analysis tool.

            ### INSTRUCTIONS
                - You will receive a list of tables in JSON format with the following structure:
                    [
                        {
                            "table_id": "Table_1",
                            "columns": [],
                            "page_range": [39, 39],
                            "rows": [
                                [
                                    "Music",
                                    "John Williams \u2022 \"I Hate Myself for Loving You\" \u2022 Joan Jett \u2022 \"Somethin' Bad\" \u2022 Miranda Lambert (Faith Hill \u2022 Pink \u2022 Carrie Underwood)"
                                ],
                                [
                                    "NFL Championship",
                                    "1955 \u2022 1956 \u2022 1957 \u2022 1958 \u2022 1959 \u2022 1960 \u2022 1961 \u2022 1962 \u2022 1963"
                                ],
                                [
                                    "AFL Championship",
                                    "1965 \u2022 1966 \u2022 1967 \u2022 1968 \u2022 1969"
                                ],
                                [
                                    "Pre-AFL-NFL merger",
                                    "I (1966) \u2022 III (1968)"
                                ],
                                [
                                    "AFC package carrier\n(1970\u20131997)",
                                    "V (1970) \u2022 VII (1972) \u2022 IX (1974) \u2022 XI (1976) \u2022 XIII (1978) \u2022 XV (1980) \u2022 XVII (1982) \u2022 XX (1985) \u2022 XXIII (1988) \u2022 XXVII (1992) \u2022 XXVIII (1993) \u2022 XXX (1995) \u2022 XXXII (1997)"
                                ],
                                [
                                    "Sunday Night Football era\n(2006\u2013present)",
                                    "XLIII (2008) \u2022 XLVI (2011) \u2022 XLIX (2014) \u2022 LII (2017) \u2022 LVI (2021) \u2022 LX (2025) \u2022 LXIV (2029)"
                                ],
                                [
                                    "Halftime shows",
                                    "V (1970) \u2022 XX (1985) \u2022 XXIII (1988) \u2022 XXVII (1992) \u2022 XXX (1995) \u2022 XLVI (2011) \u2022 XLIX (2014) \u2022 LII (2017) \u2022 LVI (2021)"
                                ],
                                [
                                    "Pro Bowl",
                                    "1952 \u2022 1953 \u2022 1958 \u2022 1959 \u2022 1960 \u2022 1961 \u2022 1962 \u2022 1963 \u2022 1964 \u2022 1965 \u2022 1972 \u2022 1974 \u2022 2009 \u2022 2012 \u2022 2013 \u2022 2014"
                                ],
                                [
                                    "NFL Honors",
                                    "2012 \u2022 2015 \u2022 2018 \u2022 2023"
                                ]
                            ]
                        },
                    ]
                
                - Your task is for each row of a table, you give a unique description, no need to list additional numbers.
                - The description sentence must be natural and semantically complete so that the embedding models can capture it.	

            ### OUTPUT
                [
                    {
                        "table_id": "Table_1",
                        "columns": [],
                        "page_range": [39, 39],
                        "rows": [
                            "Description of Row 1",
                            "Description of Row 2",
                            "Description of Row 3"
                        ]
                    }
                ]
        """

    def get_rows_description(self, table_data: list[dict]) -> str:
        for attempt in range(3):
            table_description = self.client.models.generate_content(
                model=self.model,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=self.DESCRIPTION_SYSTEM_PROMPT,
                ),
                contents=f"Please describe the following table: {table_data}"
            )

            table_description = table_description.text
            try:
                table_description = json.loads(table_description)
                break
            except Exception as e:
                print(f"Retry parsing table description {attempt + 1}")
                if attempt == 2:
                    print(f"Failed to parse table description: {e}")
                    return None
                continue

        return table_description
    
    def process_tables(self, output_path: str):
        table_names = os.listdir(self.data_path)
        for table_name in table_names:
            table_path = os.path.join(self.data_path, table_name)
            with open(table_path, "r") as f:
                table_data = json.load(f)
            table_description = self.get_rows_description(table_data)
            with open(f"{output_path}/{table_name.split('.')[0]}.json", "w") as f:
                json.dump(table_description, f)

class PDFTableIngestor:
    """
    Handles loading and parsing PDF table data from JSON files
    """

    def __init__(self):
        pass

    def load_tables_from_json(self, json_file_path: str) -> List[Dict]:
        """
        Load parsed PDF tables from JSON file

        Args:
            json_file_path (str): Path to the JSON file containing parsed tables

        Returns:
            List[Dict]: List of table dictionaries
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                tables = json.load(f)

            # Validate table structure
            validated_tables = []
            for table in tables:
                if self._validate_table_structure(table):
                    validated_tables.append(table)
                else:
                    print(f"Warning: Invalid table structure for {table.get('table_id', 'unknown')}")

            return validated_tables

        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _validate_table_structure(self, table: Dict) -> bool:
        """Validate that table has required fields"""
        required_fields = ['table_id', 'columns', 'rows']
        return all(field in table for field in required_fields)

    def get_table_stats(self, tables: List[Dict]) -> Dict[str, Any]:
        """Get statistics about the loaded tables"""
        if not tables:
            return {"total_tables": 0}

        stats = {
            "total_tables": len(tables),
            "tables_info": []
        }

        for table in tables:
            table_info = {
                "table_id": table.get('table_id', 'unknown'),
                "columns_count": len(table.get('columns', [])),
                "rows_count": len(table.get('rows', [])),
                "page_range": table.get('page_range', []),
                "has_data": len(table.get('rows', [])) > 0
            }
            stats["tables_info"].append(table_info)

        return stats

    def extract_sample_data(self, table: Dict, max_rows: int = 10) -> Dict[str, Any]:
        """Extract sample data from a table for analysis"""
        columns = table.get('columns', [])
        rows = table.get('rows', [])

        sample_rows = rows[:max_rows] if rows else []

        return {
            "table_id": table.get('table_id', 'unknown'),
            "columns": columns,
            "sample_rows": sample_rows,
            "total_rows": len(rows),
            "page_range": table.get('page_range', [])
        }
     
class PDFTablePromptConstructor:
    """
    Constructs prompts for LLM analysis of PDF tables
    """

    def __init__(self):
        self.default_template = """
            You are a data analysis expert. Analyze the following table extracted from a PDF document and create detailed metadata.

            **Table Information:**
            - Table ID: {table_id}
            - Pages: {page_range} (from PDF document)
            - Document Context: {document_context}
            - Total Rows: {total_rows}
            - Columns Count: {columns_count}

            **Table Data:**
            ```
            {table_display}
            ```

            {column_analysis}

            **Requirements:** Create metadata in the following JSON format:
            ```json
            {{
                "table_name": "descriptive_name_for_table",
                "description": "Brief description of the table's purpose and content",
                "business_context": "Business context and how this table is typically used",
                "data_category": "category_type (e.g., 'reference', 'transactional', 'analytical', 'lookup')",
                "columns": [
                    {{
                        "name": "column_name",
                        "type": "standardized_data_type",
                        "original_type": "inferred_from_content",
                        "description": "Detailed description of what this column represents",
                        "business_meaning": "Business significance of this column",
                        "data_pattern": "pattern_or_format_description",
                        "possible_values": "possible_values_if_categorical",
                        "constraints": "data_constraints_if_any",
                        "is_key": "primary_key/foreign_key/none"
                    }}
                ],
                "relationships": [
                    {{
                        "type": "relationship_type",
                        "column": "column_name",
                        "description": "description_of_relationship",
                        "references": "what_it_might_reference"
                    }}
                ],
                "data_quality_assessment": {{
                    "completeness": "assessment_of_data_completeness",
                    "consistency": "assessment_of_data_consistency",
                    "accuracy": "assessment_of_data_accuracy",
                    "notes": "additional_quality_notes"
                }},
                "usage_recommendations": [
                    "recommendation_1",
                    "recommendation_2"
                ],
                "semantic_tags": ["tag1", "tag2", "tag3"],
                "complexity_level": "simple/moderate/complex"
            }}
            ```

            **Analysis Guidelines:**
            - Analyze cell values carefully to understand data types and patterns
            - Identify potential keys, references, and relationships
            - Consider the business context based on column names and values
            - Assess data quality based on visible patterns
            - Suggest appropriate standardized data types
            - Only return valid JSON, no additional text
            """

    def create_table_analysis_prompt(
        self,
        sample_data: Dict[str, Any],
        document_context: str = "",
        custom_template: str = None
    ) -> str:
        """
        Create analysis prompt for a single table

        Args:
            sample_data (Dict): Sample data from PDFTableIngestor
            document_context (str): Additional context about the document
            custom_template (str): Custom prompt template (optional)

        Returns:
            str: Formatted prompt for LLM
        """
        template = custom_template or self.default_template

        # Format table sample for display
        table_display = self._format_table_for_display(
            sample_data['columns'],
            sample_data['sample_rows']
        )

        # Create column analysis info
        column_analysis = self._create_column_analysis_info(
            sample_data['columns'],
            sample_data['sample_rows']
        )

        prompt = template.format(
            table_id=sample_data['table_id'],
            page_range=f"{sample_data['page_range'][0]}-{sample_data['page_range'][-1]}" if sample_data['page_range'] else "unknown",
            document_context=document_context,
            table_display=table_display,
            column_analysis=column_analysis,
            total_rows=sample_data['total_rows'],
            columns_count=len(sample_data['columns'])
        )

        return prompt

    def _format_table_for_display(self, columns: List[str], rows: List[List[str]]) -> str:
        """Format table data for clean display in prompt"""
        if not columns or not rows:
            return "Empty table"

        # Create header
        formatted_table = " | ".join(columns) + "\n"
        formatted_table += " | ".join(["-" * min(len(col), 20) for col in columns]) + "\n"

        # Add rows
        for row in rows:
            # Ensure row has same length as columns
            padded_row = row + [""] * (len(columns) - len(row))
            padded_row = padded_row[:len(columns)]  # Truncate if too long

            # Clean cell values (remove newlines, limit length)
            clean_row = [str(cell).replace('\n', ' ').strip()[:100] for cell in padded_row]
            formatted_table += " | ".join(clean_row) + "\n"

        return formatted_table

    def _create_column_analysis_info(self, columns: List[str], rows: List[List[str]]) -> str:
        """Create detailed column analysis information"""
        if not columns or not rows:
            return "No data available for analysis"

        analysis = "**Column Analysis:**\n"

        for i, col in enumerate(columns):
            # Extract column values
            col_values = []
            for row in rows:
                if i < len(row) and row[i]:
                    col_values.append(str(row[i]).strip())

            # Analyze column characteristics
            unique_values = list(set(col_values))[:5]  # First 5 unique values
            has_numbers = any(self._contains_number(val) for val in col_values)
            has_dates = any(self._contains_date_pattern(val) for val in col_values)

            analysis += f"- **{col}**: "
            analysis += f"Sample values: {unique_values}, "
            analysis += f"Contains numbers: {has_numbers}, "
            analysis += f"Contains dates: {has_dates}\n"

        return analysis

    def _contains_number(self, text: str) -> bool:
        """Check if text contains numeric values"""
        import re
        return bool(re.search(r'\d+', text))

    def _contains_date_pattern(self, text: str) -> bool:
        """Check if text contains date patterns"""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or MM/DD/YYYY
            r'\w+ \d{1,2}, \d{4}'  # Month DD, YYYY
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)


    def create_batch_analysis_prompt(
        self,
        all_sample_data: List[Dict[str, Any]],
        document_context: str = ""
    ) -> str:

        """Create a prompt for analyzing multiple tables together"""
        prompt = f"""
        You are a data analysis expert. Analyze the following tables extracted from a PDF document and identify relationships between them.

        **Document Context:** {document_context}
        **Total Tables:** {len(all_sample_data)}

        """

        for i, sample_data in enumerate(all_sample_data, 1):
            table_display = self._format_table_for_display(
                sample_data['columns'],
                sample_data['sample_rows'][:5]  # Limit to 5 rows for batch analysis
            )

            prompt += f"""
            **Table {i}: {sample_data['table_id']}**
            ```
            {table_display}
            ```

            """

        prompt += """
        **Requirements:** Analyze the relationships between these tables and provide:
        1. Potential foreign key relationships
        2. Data flow between tables
        3. Business process representation
        4. Recommended table usage order
        5. Data integration opportunities

        Provide analysis in structured text format.
        """

        return prompt

class PDFTableMetadataGenerator:
    """
    Generates metadata for PDF tables using LLM analysis
    """

    def __init__(self):
        """
        Initialize metadata generator

        """

        self.llm_service = LLMService()
        self.providers = [{"name": "gemini", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.0, "retry": 3}]
        # self.providers = [{"name": "gemini", "model": "gemini-2.0-flash", "temperature": 0.0, "retry": 3}]
        self.ingestor = PDFTableIngestor()
        self.prompt_constructor = PDFTablePromptConstructor()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_single_table_metadata(
        self,
        sample_data: Dict[str, Any],
        document_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate metadata for a single table

        Args:
            sample_data (Dict): Sample data from PDFTableIngestor
            document_context (str): Document context

        Returns:
            Dict: Generated metadata
        """
        try:
            # Create analysis prompt
            prompt = self.prompt_constructor.create_table_analysis_prompt(
                sample_data, document_context
            )

            # # Call LLM
            # response = self.llm_service.complete(
            #     system_prompt=prompt,
            #     user_prompt="Let's generate metadata for the table",
            #     json_output=True,
            #     providers=self.providers,
            # )
            for attempt in range(5):
                response = self.client.models.generate_content(
                    contents=[prompt],
                    model="gemini-2.5-flash-preview-04-17",
                    config=GenerateContentConfig(response_mime_type="application/json")
                )
                
                try:
                    metadata = json.loads(response.text)
                except json.JSONDecodeError as jde:
                    print(f"JSON decode error: {jde}")
                    if attempt == 4:
                        print(f"Failed to generate metadata for {sample_data['table_id']} after {attempt + 1} attempts")
                        metadata = None
                    continue
                break

            # # Parse response
            # metadata = self._parse_llm_response(response)

            # metadata = response

            # Add source information
            metadata['source_info'] = {
                'table_id': sample_data['table_id'],
                'page_range': sample_data['page_range'],
                'total_rows': sample_data['total_rows'],
                'columns_count': len(sample_data['columns']),
                'original_columns': sample_data['columns']
            }
            metadata['generated_at'] = datetime.now().isoformat()

            return metadata

        except Exception as e:
            print(f"Error generating metadata for {sample_data['table_id']}: {e}")
            return self._create_fallback_metadata(sample_data)

    def generate_all_tables_metadata(
        self,
        json_file_path: str,
        document_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate metadata for all tables in a PDF JSON file

        Args:
            json_file_path (str): Path to PDF tables JSON file
            document_context (str): Document context

        Returns:
            Dict: Complete metadata for all tables
        """
        # Load tables
        tables = self.ingestor.load_tables_from_json(json_file_path)
        table_stats = self.ingestor.get_table_stats(tables)

        # Initialize result structure
        all_metadata = {
            'document_info': {
                'source_file': json_file_path,
                'context': document_context,
                'total_tables': len(tables),
                'processed_at': datetime.now().isoformat(),
                'statistics': table_stats
            },
            'tables': {}
        }

        print(f"Processing {len(tables)} tables...")

        # Process each table
        for table in tables:
            table_id = table['table_id']
            print(f"Processing {table_id}...")

            try:
                # Extract sample data
                sample_data = self.ingestor.extract_sample_data(table)

                # Generate metadata
                metadata = self.generate_single_table_metadata(sample_data, document_context)
                all_metadata['tables'][table_id] = metadata

                print(f"✅ Successfully processed {table_id}")

            except Exception as e:
                print(f"❌ Error processing {table_id}: {e}")
                all_metadata['tables'][table_id] = self._create_fallback_metadata(
                    self.ingestor.extract_sample_data(table)
                )

        return all_metadata

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            raise

    def _create_fallback_metadata(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic metadata when LLM analysis fails"""
        columns = sample_data['columns']

        column_metadata = []
        for col in columns:
            column_metadata.append({
                'name': col,
                'type': 'string',
                'description': f'Column {col} from table {sample_data["table_id"]}',
                'business_meaning': 'Not analyzed',
                'data_pattern': 'Unknown'
            })

        return {
            'table_name': sample_data['table_id'],
            'description': 'Auto-generated basic metadata from PDF table',
            'columns': column_metadata,
            'source_info': {
                'table_id': sample_data['table_id'],
                'total_rows': sample_data['total_rows'],
                'columns_count': len(columns),
                'original_columns': columns
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'fallback'
        }
    
class TableTextFormatter:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()
        self.tables = self.data.get("tables", {})

    def _load_json(self):
        with open(self.json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def format_table(self, table):
        table_name = table.get("table_name", "Unknown Table")
        description = table.get("description", "No description available.")
        columns = table.get("columns", [])

        result = f"Table: {table_name}\n"
        result += f"Purpose: {description}\n"
        result += "Columns:\n"

        for col in columns:
            name = col.get("name", "Unnamed")
            dtype = col.get("type", "Unknown")
            desc = col.get("description", "No description.")
            result += f"- {name} ({dtype}): {desc}\n"

        return result

    def format_all_tables(self):
        formatted_tables = {}
        for table_id, table in self.tables.items():
            formatted_tables[table_id] = self.format_table(table)
        return formatted_tables

    def save_to_file(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        formatted_tables = self.format_all_tables()
        base_filename = os.path.splitext(os.path.basename(self.json_path))[0]

        for table_id, table_content in formatted_tables.items():
            save_path = os.path.join(output_path, f"{base_filename}_{table_id}.txt")
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(table_content)
            print(f"Formatted table saved to: {save_path}")
