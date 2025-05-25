import os
from src.services.processing.table_processing import extract_table_data
import json

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

if __name__ == "__main__":
    json_paths = os.listdir("parsing/llm_parsing")

    pdf_paths = group_by_prefix(json_paths)
    for pdf_path in pdf_paths:
        full_path = [os.path.join("parsing/llm_parsing", path) for path in pdf_path]
        tb_data = extract_table_data(full_path)

        pdf_name = pdf_path[0].split("_")[0]
        os.makedirs(f"m3docvqa/tables_dev/", exist_ok=True)

        with open(f"m3docvqa/tables_dev/{pdf_name}.json", "w") as f:
            json.dump(tb_data, f)