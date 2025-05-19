import csv
import json
import os
from src.services.llm import LLMService

llm_service = LLMService()

providers = [{"name": "gemini", "model": "gemini-2.5-flash-preview-04-17", "temperature": 0.9, "retry": 3}]

MERGE_SYSTEM_PROMPT = """
    ### ROLE
        - You are a table processing tool.

    ### INSTRUCTIONS
        - You will receive a dictionary containing information about the tables with:
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
"""

def extract_table_data(data_paths):
    # Nhận vào toàn bộ paths json parsing từ 1 tài liệu -> lấy ra hết table -> gửi lên LLM -> lấy kết quả rồi gộp table lại.

    small_tables = {}
    tables = {}
    for data_path in data_paths:
        with open(data_path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if "table" in key:
                tb_key = os.path.basename(data_path).split(".")[0] + "." + key
                small_tables[tb_key] = get_small_table(value)
                tables[tb_key] = value

    merge_info = llm_service.complete(
        system_prompt=MERGE_SYSTEM_PROMPT,
        user_prompt=json.dumps(small_tables),
        json_output=True,
        providers=providers,
    )
    
    for key, value in merge_info.items():
        tb_keys = key.split("-")
        merge_value = ""
        for tb_key in tb_keys:
            merge_value += f"\n{tables[tb_key]}"
            del tables[tb_key]
        tables[key] = merge_value

    return tables

def get_small_table(table_data: str) -> str:
    t_reader = csv.reader(table_data.split("\n"))

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