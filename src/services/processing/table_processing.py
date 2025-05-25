import csv
import json
import os
from src.services.llm import LLMService
from loguru import logger
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
    
    
    
    
