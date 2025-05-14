import argparse
import os
import numpy as np
import random
from src.db import chroma_client
from src.db.milvus import milvus_colbert_retriever
from src.services.llm import LLMService
import glob
import jsonlines
from src.services.eval import MultiModalEval

COLLECTION_NAME = "m3docvqa_500"

RAG_PROMPT = """
You are a helpful assistant that can answer questions about the image.
If the question is not related to the image, please answer "I don't know".
"""

providers = [{"name": "gemini-image", "model": "gemini-1.5-pro", "temperature": 0.9, "retry": 3}]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_embedding_folder", type=str, required=False, default="m3docvqa/question_embeddings")
    parser.add_argument("--qa_file", type=str, required=False, default="m3docvqa/multimodalqa/MMQA_dev.jsonl")
    parser.add_argument("--num_question", type=int, required=False, default=1)
    parser.add_argument("--image_folder", type=str, required=False, default="m3docvqa/images_dev")
    parser.add_argument("--db", type=str, required=False, default="milvus")
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--output_file", type=str, required=False, default="eval_results.jsonl")
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

    collection = chroma_client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=topk,
    )
    image_paths = [result["image_name"] for result in results["metadatas"][0]]
    return image_paths

def use_milvus_db(question_embedding_file: str, topk: int):
    question_embedding = np.load(question_embedding_file)
    question_embedding = np.squeeze(question_embedding, axis=0)

    results = milvus_colbert_retriever.search(question_embedding, topk)

    image_paths = []
    for result in results:
        doc_id = result[1]
        res = milvus_colbert_retriever.client.query(    
            collection_name="m3docvqa_500",
            filter=f"doc_id == {doc_id}",
            output_fields=["doc"]
        )
        image_paths.append(os.path.basename(res[0]["doc"]))

    return image_paths

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
    
    questions = []
    predictions = []
    ground_truths = []
    retrieval_context = []

    while sample_idx < args.num_question and i < len(question_embedding_files):
        question_embedding_file = question_embedding_files[i]
        question, answers, support_context = mapping_question(question_embedding_file, args.qa_file)
        
        for sp in support_context:
            if sp["doc_id"] in doc_ids:
                question_embedding = np.load(question_embedding_file)
                question_embedding = np.squeeze(question_embedding, axis=0)
                question_embedding = np.mean(question_embedding, axis=0)
                question_embedding = question_embedding.tolist()

                if args.db == "chroma":
                    related_image_names = use_chroma_db(question_embedding_file, topk=topk)
                elif args.db == "milvus":
                    related_image_names = use_milvus_db(question_embedding_file, topk=topk)
                
                related_image_paths = [os.path.join(image_folder, image_name) for image_name in related_image_names]

                llm_answer = llm_service.complete(
                    system_prompt=RAG_PROMPT,
                    user_prompt=question,
                    image_paths=related_image_paths,
                    providers=providers,
                )

                questions.append(question)
                predictions.append([llm_answer])
                ground_truths.append([answers[0]["answer"]])
                retrieval_context.append(related_image_paths)

                sample_idx += 1
                break
        i += 1
        
    eval_service = MultiModalEval(questions, predictions, ground_truths, retrieval_context)
    test_cases = eval_service.make_test_case()
    results = eval_service.evaluate(test_cases)
    
    with open(output_file, "w") as f:
        for i in range(len(questions)):
            line = {
                "question": questions[i],
                "prediction": predictions[i],
                "ground_truth": ground_truths[i],
                "retrieval_context": retrieval_context[i],
                "eval": results[i]
            }
            jsonlines.Writer(f).write(line)
if __name__ == "__main__":
    main()