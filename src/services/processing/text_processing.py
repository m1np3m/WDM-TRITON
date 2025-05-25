import json

def processing_dict_data(dict_data: dict) -> str:
    data = ""
    for _key, value in dict_data.items():
        data += str(value) + "\n"
    return data

def merge_json_data(json_paths: list[str]) -> str:
    data = ""
    for json_path in json_paths:
        with open(json_path, "r") as f:
            try:
                data += processing_dict_data(json.load(f)) + "\n\n"
            except Exception as e:
                print(f"Error format {json_path}: {e}")
                data += str(f.read()) + "\n\n"
                continue
    return data



