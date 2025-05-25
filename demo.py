import argparse
import os
import numpy as np
import random
from src.db import chroma_client
from src.db.milvus import MilvusColbertRetriever, MilvusBgeM3Retriever
from src.services.llm import LLMService
from src.models.embedding.bge_m3 import BgeM3MilvusEmbedding
import glob
import jsonlines
from pymilvus import MilvusClient

COPALI_COLLECTION_NAME = "m3docvqa_copali"
TEXT_COLLECTION_NAME = "m3docvqa_text"

IMAGE_RAG_PROMPT = """
You are a helpful and intelligent assistant. Below is contextual information extracted from one or more images provided by the user. This may include OCR text, image captions, object descriptions, or metadata.
Based on this image-derived context, answer the user's question as accurately and concisely as possible.
"""

TEXT_RAG_PROMPT = """
You are a helpful assistant that can answer questions about the text.
If the question is not related to the text, please answer "I don't know".
"""

providers = [{"name": "gemini-image", "model": "gemini-2.0-flash", "temperature": 0.9, "retry": 3}]
milvus_client = MilvusClient(uri="milvus_db/milvus.db")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_embedding_folder", type=str, required=False, default='m3docvqa/question_embeddings')
    parser.add_argument("--qa_file", type=str, required=False, default='m3docvqa/multimodalqa/MMQA_dev.jsonl')
    parser.add_argument("--num_question", type=int, required=False, default=1)
    parser.add_argument("--image_folder", type=str, required=False, default='m3docvqa/images_dev')
    parser.add_argument("--output_file", type=str, required=False, default='demo_results.jsonl')
    parser.add_argument("--db", type=str, required=False, default='copali_milvus')
    parser.add_argument("--topk", type=int, required=False, default=5)
    return parser.parse_args()

def mapping_question(question_embedding_file: str, qa_file: str):
    qid = os.path.basename(question_embedding_file).split(".")[0]
    with jsonlines.open(qa_file, "r") as reader:
        for qa in reader:
            if qa["qid"] == qid:
                return qa["question"], qa["answers"], qa["supporting_context"]

def use_chroma_db(question_embedding_file: str, topk: int):
    question_embedding = np.load(question_embedding_file)
    question_embedding = np.squeeze(question_embedding, axis=0)
    question_embedding = np.mean(question_embedding, axis=0)
    question_embedding = question_embedding.tolist()

    collection = chroma_client.get_collection(COPALI_COLLECTION_NAME)
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=topk,
    )
    image_paths = [result["image_name"] for result in results["metadatas"][0]]
    return image_paths

def use_copali_milvus_db(question_embedding_file: str, topk: int):
    question_embedding = np.load(question_embedding_file)
    question_embedding = np.squeeze(question_embedding, axis=0)

    milvus_colbert_retriever = MilvusColbertRetriever(milvus_client, COPALI_COLLECTION_NAME)
    results = milvus_colbert_retriever.search(question_embedding, topk)

    image_paths = []
    for result in results:
        doc_id = result[1]
        res = milvus_colbert_retriever.client.query(    
            collection_name=COPALI_COLLECTION_NAME,
            filter=f"doc_id == {doc_id}",
            output_fields=["doc"]
        )
        image_paths.append(os.path.basename(res[0]["doc"]))

    return image_paths

def use_bge_m3_milvus_db(question: str, topk: int):
    bge_m3_embedding = BgeM3MilvusEmbedding()
    bge_m3_retriever = MilvusBgeM3Retriever(milvus_client, TEXT_COLLECTION_NAME)
    question_embedding = bge_m3_embedding.encode([question])
    dense_vector = question_embedding["dense"][0]

    sparse_vector = question_embedding["sparse"][0]
    sparse_vector = {c: v for c, v in zip(sparse_vector.col, sparse_vector.data)}

    dense_results = bge_m3_retriever.dense_search(dense_vector, topk)
    # sparse_results = bge_m3_retriever.sparse_search(sparse_vector, topk)

    text_results = [text_result["text"] for text_result in dense_results]
    return text_results

def main():
    args = parse_args()
    question_embedding_folder = args.question_embedding_folder
    image_folder = args.image_folder
    output_file = args.output_file
    topk = args.topk
    llm_service = LLMService()

    image_names = os.listdir(image_folder)
    doc_ids = [image_name.split(".")[0].split("_")[0] for image_name in image_names]
    doc_ids = list(set(doc_ids))
    print(f"Total number of documents: {len(doc_ids)}")
    question_embedding_files = glob.glob(os.path.join(question_embedding_folder, "*.npy"))
    random.shuffle(question_embedding_files)

    i = 0
    sample_idx = 0
    
    output_data = []
    while sample_idx < args.num_question and i < len(question_embedding_files):
        question_embedding_file = question_embedding_files[i]
        question, answers, support_context = mapping_question(question_embedding_file, args.qa_file)
        
        related_image_paths = []
        related_text = []

        for sp in support_context:
            if sp["doc_id"] in doc_ids:

                if args.db == "chroma":
                    related_image_names = use_chroma_db(question_embedding_file, topk=topk)
                    related_image_paths = [os.path.join(image_folder, image_name) for image_name in related_image_names]
                    llm_answer = llm_service.complete(
                        system_prompt=IMAGE_RAG_PROMPT,
                        user_prompt=question,
                        file_paths=related_image_paths,
                        providers=providers,
                    )
                elif args.db == "copali_milvus":
                    related_image_names = use_copali_milvus_db(question_embedding_file, topk=topk)
                    related_image_paths = [os.path.join(image_folder, image_name) for image_name in related_image_names]
                    llm_answer = llm_service.complete(
                        system_prompt=IMAGE_RAG_PROMPT,
                        user_prompt=f"Question: {question}",
                        file_paths=related_image_paths,
                        providers=providers,
                    )
                elif args.db == "bge_m3_milvus":
                    related_text = use_bge_m3_milvus_db(question, topk=topk)
                    llm_answer = llm_service.complete(
                        system_prompt=TEXT_RAG_PROMPT,
                        user_prompt=f"Question: {question}\nRelated text: {related_text}",
                        providers=providers,
                    )

                output_data.append({
                    "question": question,
                    "answers": answers,
                    "llm_answer": llm_answer,
                    "related_image_paths": related_image_paths,
                    "related_text": related_text,
                })

                sample_idx += 1
                break
        i += 1
        
    with open(output_file, "w") as f:
        for data in output_data:
            jsonlines.Writer(f).write(data)

if __name__ == "__main__":
    main()
        

    








