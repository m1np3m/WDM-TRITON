import argparse
import os
import numpy as np
import random
from src.db import chroma_client
from src.services.llm import LLMService
import glob
import jsonlines
COLLECTION_NAME = "m3docvqa"

RAG_PROMPT = """
You are a helpful assistant that can answer questions about the image.
If the question is not related to the image, please answer "I don't know".
"""

providers = [{"name": "gemini-image", "model": "gemini-1.5-pro", "temperature": 0.9, "retry": 3}]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_embedding_folder", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--num_question", type=int, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()

def mapping_question(question_embedding_file: str, qa_file: str):
    qid = os.path.basename(question_embedding_file).split(".")[0]
    with jsonlines.open(qa_file, "r") as reader:
        for qa in reader:
            if qa["qid"] == qid:
                return qa["question"], qa["answers"]

def main():
    args = parse_args()
    question_embedding_folder = args.question_embedding_folder
    image_folder = args.image_folder
    output_folder = args.output_folder

    llm_service = LLMService()

    question_embedding_files = glob.glob(os.path.join(question_embedding_folder, "*.npy"))
    question_embedding_files = random.sample(question_embedding_files, args.num_question)

    output_data = []
    for question_embedding_file in question_embedding_files:
        question, answers = mapping_question(question_embedding_file, args.qa_file)

        question_embedding = np.load(question_embedding_file)
        question_embedding = np.squeeze(question_embedding, axis=0)
        question_embedding = np.mean(question_embedding, axis=0)
        question_embedding = question_embedding.tolist()

        collection = chroma_client.get_collection(COLLECTION_NAME)
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=5,
        )

        related_image_names = [result["image_name"] for result in results["metadatas"][0]]
        related_image_paths = [os.path.join(image_folder, image_name) for image_name in related_image_names]

        llm_answer = llm_service.complete(
            system_prompt=RAG_PROMPT,
            user_prompt=question,
            image_paths=related_image_paths,
            providers=providers,
        )

        output_data.append({
            "question": question,
            "answers": answers,
            "llm_answer": llm_answer,
            "related_image_paths": related_image_paths,
        })
        
    with open(os.path.join(output_folder, "output.jsonl"), "w") as f:
        for data in output_data:
            jsonlines.Writer(f).write(data)


if __name__ == "__main__":
    main()
        

    








