import argparse
import json
import os
import numpy as np
import random
from src.db import chroma_client
from src.db.milvus import MilvusColbertRetriever
from src.services.llm import LLMService
import glob
import jsonlines
from src.services.eval import MultiModalEval
from pymilvus import MilvusClient
from src.models.embedding.bge_m3 import BgeM3MilvusEmbedding
from src.db.milvus import MilvusBgeM3Retriever

COLLECTION_NAME = "m3docvqa_copali"
TEXT_COLLECTION_NAME = "m3docvqa_text"

IMAGE_RAG_PROMPT = """
You are a helpful assistant that can answer questions about the image.
If the question is not related to the image, please answer "I don't know".
"""

TEXT_RAG_PROMPT = """
You are a helpful assistant that can answer questions about the text.
If the question is not related to the text, please answer "I don't know".
"""

RAG_PROMPT = """
You are an intelligent assistant designed to process and analyze documents provided as either images or text. 

1. If the input is an image, first extract the text content from the image using OCR.
2. Then, analyze the extracted text or directly analyze the given text input.
3. Based on the content, provide a concise and informative summary or answer any specific questions related to the document.
4. Ensure your output is clear, accurate, and relevant to the input content.
"""
providers = [{"name": "gemini-image", "model": "gemini-2.0-flash", "temperature": 0.9, "retry": 3}]

milvus_client = MilvusClient(uri="milvus_db/milvus.db")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_embedding_folder", type=str, required=False, default="m3docvqa/question_embeddings")
    parser.add_argument("--qa_folder", type=str, required=False, default="m3docvqa/multimodalqa/tables/")
    parser.add_argument("--num_question", type=int, required=False, default=10)
    parser.add_argument("--image_folder", type=str, required=False, default="m3docvqa/images_dev")
    parser.add_argument("--db", type=str, required=False, default="bge_m3_milvus")
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--output_file", type=str, required=False, default="eval_results.jsonl")
    parser.add_argument("--step", type=str, required=False, default="retrieval")
    return parser.parse_args()

def make_qa_test(qa_folder: str, num_question: int):
    qa_files = os.listdir(qa_folder)

    test_cases = []
    for qa_file in qa_files:
        qa_file = os.path.join(qa_folder, qa_file)
        with open(qa_file, "r") as f:
            qa = json.load(f)
            for i in range(len(qa)):
                gts = [os.path.basename(qa_file).split(".")[0] + "_" + str(page) for page in qa[i]["pages"]]
                test_cases.append({
                    "qid": os.path.basename(qa_file).split(".")[0] + "_" + str(i),
                    "question": qa[i]["question"],
                    "answers": qa[i]["answer"],
                    "ground_truths": gts
                })
    random.shuffle(test_cases)
    return test_cases[:min(num_question, len(test_cases))]

def use_chroma_db(retriever, question_embedding_file: str, topk: int):
    question_embedding = np.load(question_embedding_file)
    question_embedding = np.squeeze(question_embedding, axis=0)
    question_embedding = np.mean(question_embedding, axis=0)
    question_embedding = question_embedding.tolist()

    results = retriever.query(
        query_embeddings=question_embedding,
        n_results=topk,
    )
    image_paths = [result["image_name"] for result in results["metadatas"][0]]
    return image_paths

def use_copali_milvus_db(retriever, question_embedding_file: str, topk: int):
    question_embedding = np.load(question_embedding_file)
    question_embedding = np.squeeze(question_embedding, axis=0)

    results = retriever.search(question_embedding, topk)

    image_paths = []
    for result in results:
        doc_id = result[1]
        res = retriever.client.query(    
            collection_name=COLLECTION_NAME,
            filter=f"doc_id == {doc_id}",
            output_fields=["doc"]
        )
        image_paths.append(os.path.basename(res[0]["doc"]))

    return image_paths

def use_bge_m3_milvus_db(embedding, retriever, question: str, topk: int):
    question_embedding = embedding.encode([question])
    dense_vector = question_embedding["dense"][0]

    sparse_vector = question_embedding["sparse"][0]
    sparse_vector = {c: v for c, v in zip(sparse_vector.col, sparse_vector.data)}

    # dense_results = bge_m3_retriever.dense_search(dense_vector, topk)
    # sparse_results = bge_m3_retriever.sparse_search(sparse_vector, topk)

    hybrid_results = retriever.hybrid_search(dense_vector, sparse_vector, topk=topk)

    text_results = [result["text"] for result in hybrid_results]
    doc_ids = [result["doc_id"] for result in hybrid_results]

    return text_results, doc_ids

def main():
    args = parse_args()
    question_embedding_folder = args.question_embedding_folder
    image_folder = args.image_folder
    output_file = args.output_file
    topk = args.topk
    
    llm_service = LLMService()

    if args.db == "chroma":
        retriever = chroma_client.get_collection(COLLECTION_NAME)
    elif args.db == "copali_milvus":
        retriever = MilvusColbertRetriever(milvus_client, COLLECTION_NAME)
    elif args.db == "bge_m3_milvus":
        embedding = BgeM3MilvusEmbedding()
        retriever = MilvusBgeM3Retriever(milvus_client, TEXT_COLLECTION_NAME)

    image_names = os.listdir(image_folder)
    doc_ids = [image_name.split(".")[0].split("_")[0] for image_name in image_names]
    doc_ids = list(set(doc_ids))
    print(f"Total number of documents: {len(doc_ids)}")

    questions = []
    predictions = []
    retrieval_preds = []
    ground_truths = []
    retrieval_gts = []
    retrieval_context = []

    qa_test = make_qa_test(args.qa_folder, args.num_question)

    for i in range(len(qa_test)):
        question_embedding_file = os.path.join(question_embedding_folder, qa_test[i]["qid"] + ".npy")
        question, answer, retrieval_gt = qa_test[i]["question"], qa_test[i]["answers"], qa_test[i]["ground_truths"]
        
        related_image_names = []
        related_doc_ids = []
        related_text = []
        llm_answer = ""
        
        if args.db == "chroma":
            related_image_names = use_chroma_db(retriever, question_embedding_file, topk=topk)
            related_doc_ids = [image_name.split(".")[0] for image_name in related_image_names]
        elif args.db == "copali_milvus":
            related_image_names = use_copali_milvus_db(retriever, question_embedding_file, topk=topk)
            related_doc_ids = [image_name.split(".")[0] for image_name in related_image_names]
        elif args.db == "bge_m3_milvus":
            related_text, related_doc_ids = use_bge_m3_milvus_db(embedding, retriever, question, topk=topk)

        related_image_paths = [os.path.join(image_folder, image_name) for image_name in related_image_names]
        
        if args.step == "rag":
            llm_answer = llm_service.complete(
                system_prompt=RAG_PROMPT,
                user_prompt=f"Please answer the user's question: {question} based on the DOCUMENT and RETRIEVED CHUNKS: {related_text}",
                file_paths=related_image_paths,
                providers=providers,
            )
            
        questions.append(question)
        predictions.append([llm_answer])
        retrieval_preds.append(related_doc_ids)
        ground_truths.append([answer])
        retrieval_gts.append(retrieval_gt)
        retrieval_context.append(related_text + related_image_paths)
        
    if args.step == "rag":
        eval_service = MultiModalEval(questions, predictions, ground_truths, retrieval_context)
        test_cases = eval_service.make_test_case()
        results = eval_service.evaluate(test_cases)
    elif args.step == "retrieval":
        eval_service = MultiModalEval(questions, retrieval_preds, retrieval_gts)
        results = eval_service.evaluate_retrieval(k=topk)
    
    with open(output_file, "w") as f:
        for i in range(len(questions)):
            line = {
                "question": questions[i],
                "prediction": predictions[i],
                "ground_truth": ground_truths[i],
                "retrieval_pred": retrieval_preds[i],
                "retrieval_gt": retrieval_gts[i],
                "retrieval_context": retrieval_context[i],
                "eval": results[i]
            }
            jsonlines.Writer(f).write(line)
        
        jsonlines.Writer(f).write(results[-1])
            
if __name__ == "__main__":
    main()