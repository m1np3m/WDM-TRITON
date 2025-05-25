import json
import os
from src.services.llm import LLMService

class GenerateTableQuestionsService:
    def __init__(self, num_questions: int, data_path: str):
        self.llm_service = LLMService()
        self.num_questions = num_questions
        self.data_path = data_path

        self.providers = [{"name": "gemini-file", "model": "gemini-2.5-pro-preview-05-06", "temperature": 0.9, "retry": 3}]

        self.QUESTION_SYSTEM_PROMPT = """
        ### ROLE
	    - You are a QA generation assistant. Your task is to create high-quality question–answer (Q&A) pairs to evaluate a Retrieval-Augmented Generation (RAG) system using a large tabular dataset that spans multiple pages.

        ### INSTRUCTIONS
            - The table is fragmented across several document pages, and answering each question may require combining information from different pages.
            - Generate a diverse set of questions based on the table with the following criteria:
                    + Cross-row reasoning (e.g. trends, comparisons)
                    + Cross-page reasoning (data from multiple pages needed)
                    + Aggregation (e.g. totals, averages)
                    + Filtering by column values (e.g. "find all rows with X")

        - For each question, include:
            + A concise, accurate answer derived from the table
            + An array of page numbers needed to answer the question
            + A flag "requires_multi_chunk" = true if data spans more than one page

        ### OUTPUT
            - Format the output in JSON like this:
            [
            {
                "question": "What is the average sales revenue of Category A between 2020 and 2023?",
                "answer": "$135,000",
                "pages": [3, 4],
                "requires_multi_chunk": true
            },
            {
                "question": "Which product had the highest revenue in 2022?",
                "answer": "Product X",
                "pages": [2],
                "requires_multi_chunk": false
            }
            ]
        """
    def generate_questions(self, output_path: str):
        document_names = os.listdir(self.data_path)
        for document_name in document_names:
            output_file_path = os.path.join(output_path, f"{document_name}.json")
            if os.path.exists(output_file_path):
                continue
            qa_pairs = self.llm_service.complete(
                    system_prompt=self.QUESTION_SYSTEM_PROMPT,
                    user_prompt=f"Please create me {self.num_questions} sets of questions/answers",
                    providers=self.providers,
                    file_paths=[os.path.join(self.data_path, document_name)],
                    json_output=True,
                )
            
            with open(os.path.join(output_path, f"{document_name}.json"), "w") as f:
                json.dump(qa_pairs, f)