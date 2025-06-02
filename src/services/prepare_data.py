import os
import sys
from pathlib import Path

import chromadb
import dotenv

# Add root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from src.datasets import M3DocVQA
from src.db.milvus import MilvusBgeM3Retriever, MilvusLLMRetriever, MilvusColbertRetriever
from src.db.chroma_client import ChromaRetriever
from pymilvus import MilvusClient
from argparse import ArgumentParser
from src.services.processing.table_processing import *
from src.models.parser.llm_tables_parser import LLMTableParserV2
from src.models.parser.llm_text_parser import LLMTextParser

dotenv.load_dotenv()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="m3docvqa/pdfs_dev/")
    parser.add_argument("--type", type=str, default="table")
    parser.add_argument("--db", type=str, default="milvus")
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--embedding_type", type=str, default="bge_m3")
    parser.add_argument("--collection_name", type=str, default="m3docvqa_table_metadata_description_2steps")
    parser.add_argument("--batch_size", type=int, default=100)
    return parser.parse_args()

def prepare_table_metadata_data(data_path: str):
    table_path = "data/tables"
    table_metadata_path = "data/tables_metadata"
    table_description_path = "data/tables_metadata_description"

    generator = PDFTableMetadataGenerator()

    if not os.path.exists(table_path):
        os.makedirs(table_path)

    if not os.path.exists(table_description_path):
        os.makedirs(table_description_path)

    if not os.path.exists(table_metadata_path):
        os.makedirs(table_metadata_path)

    # Parse table data from PDF using LLM
    parser = LLMTableParserV2(data_path)
    parser.parse(output_path=table_path)

    for filename in sorted(os.listdir(table_path)):
        json_file = os.path.join(table_path, filename)
        print(f"🔄 Processing {json_file}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ File {filename} is invalid or unreadable. Writing empty metadata.")
            data = None

        output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(table_metadata_path, output_filename)

        if data is None or data == {}:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
            print(f"🚫 Empty or invalid JSON. Written empty metadata to: {output_path}\n")
            continue

        document_context = ""
        all_metadata = generator.generate_all_tables_metadata(json_file, document_context)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=4)

        print(f"✅ Done! Saved to: {output_path}")
        print(f"📊 Processed {all_metadata['document_info']['total_tables']} tables\n")
    
    for filename in os.listdir(table_metadata_path):
        json_path = os.path.join(table_metadata_path, filename)
        TableTextFormatter(json_path).save_to_file(table_description_path)

def prepare_table_description_data(data_path: str):
    table_path = "data/tables"
    table_description_path = "data/tables_description"

    if not os.path.exists(table_description_path):
        os.makedirs(table_description_path)

    # # Parse table data from PDF using LLM
    # parser = LLMTableParserV2(data_path)
    # parser.parse(output_path=table_path)

    table_processor = TableRowDescription(table_path)
    table_processor.process_tables(output_path=table_description_path)

def prepare_text_data(data_path: str):
    text_path = "data/text"

    if not os.path.exists(text_path):
        os.makedirs(text_path)

    parser = LLMTextParser(data_path)
    parser.parse(output_path=text_path)

def main():
    args = parse_args()
    
    embedding_type = args.embedding_type
    collection_name = args.collection_name
    batch_size = args.batch_size


    ### Prepare data
    if args.type == "table":
        print("🔄 Preparing table data...")
        # prepare_table_metadata_data(args.data_path)
        # prepare_table_description_data(args.data_path)
        m3docvqa = M3DocVQA(data_path="data/2-steps/tables_metadata_description")
    
    elif args.type == "text":
        print("🔄 Preparing text data...")
        prepare_text_data(data_path=args.data_path)
        m3docvqa = M3DocVQA(data_path="data/text")

    elif args.type == "multimodal_copali":
        print("🔄 Preparing multimodal data...")
        m3docvqa = M3DocVQA(data_path=args.data_path, embedding_path=args.embedding_path)
    

    ### Prepare database
    if args.type == "multimodal_copali":
        if args.db == "milvus":
            milvus_client = MilvusClient(uri=os.getenv("MILVUS_DB_PATH"))
            milvus_colbert_retriever = MilvusColbertRetriever(milvus_client, collection_name)
            milvus_colbert_retriever.create_collection()
            milvus_colbert_retriever.create_index()
            m3docvqa.add_copali_data_to_milvus_db(milvus_client, collection_name, batch_size = 1)

        elif args.db == "chroma":
            chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
            chroma_retriever = ChromaRetriever(chroma_client, collection_name)
            chroma_retriever.create_collection()
            m3docvqa.add_copali_data_to_chroma_db(chroma_client, collection_name, batch_size = 1)

    elif args.type == "table":
        milvus_client = MilvusClient(uri=os.getenv("MILVUS_DB_PATH"))
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            milvus_bge_m3_retriever.create_collection()
            milvus_bge_m3_retriever.create_index()
            m3docvqa.add_table_metadata_to_milvus_db(milvus_client, collection_name, "data/2-steps/tables", batch_size, embedding_type)
            # m3docvqa.add_table_description_to_milvus_db(milvus_client, collection_name, 1, embedding_type)
        
        elif embedding_type == "llm":
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            milvus_llm_retriever.create_collection()
            milvus_llm_retriever.create_index()
            # m3docvqa.add_table_metadata_to_milvus_db(milvus_client, collection_name, 'data/tables_metadata', batch_size, embedding_type)
            m3docvqa.add_table_description_to_milvus_db(milvus_client, collection_name, 1, embedding_type)
    
    elif args.type == "text":
        milvus_client = MilvusClient(uri=os.getenv("MILVUS_DB_PATH"))
        if embedding_type == "bge_m3":
            milvus_bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, collection_name)
            milvus_bge_m3_retriever.create_collection()
            milvus_bge_m3_retriever.create_index()
            m3docvqa.add_text_data_to_milvus_db(milvus_client, collection_name, batch_size, embedding_type)
        
        elif embedding_type == "llm":
            milvus_llm_retriever = MilvusLLMRetriever(milvus_client, collection_name)
            milvus_llm_retriever.create_collection()
            milvus_llm_retriever.create_index()
            m3docvqa.add_text_data_to_milvus_db(milvus_client, collection_name, batch_size, embedding_type)
            
if __name__ == "__main__":
    main()
