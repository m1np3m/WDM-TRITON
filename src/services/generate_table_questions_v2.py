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
            - You are a QA generation assistant. Your task is to create high-quality question-answer (Q&A) pairs to evaluate a Retrieval-Augmented Generation (RAG) system using a large tabular dataset that spans multiple pages.

        ### INSTRUCTIONS
            - You will be given a json that contains a list of tables of a file and will generate a diverse set of questions based on these tables with the following criteria:
                + Cross-row reasoning (e.g. trends, comparisons)
                + Aggregation (e.g. totals, averages)
                + Filtering by column values (e.g. "find all rows with X")
                + Questions should be varied in complexity and type from asking some information to comparing different values
            - Questions must just use the information provided in the tables and should not require external knowledge
            - For each question, include:
                + A concise, accurate answer derived from the table
                + ID of the table from which the question is derived

            ### OUTPUT
            - Format the output in JSON like this:
            [
                {
                    "question": "What is the average sales revenue of Category A between 2020 and 2023?",
                    "answer": "$135,000",
                    "table_id": "Table_1"
                },
                {
                    "question": "Which product had the highest revenue in 2022?",
                    "answer": "Product X",
                    "table_id": "Table_2"
                }
            ]
        """
    def generate_questions(self, output_path: str):
        document_names = os.listdir(self.data_path)
        for document_name in document_names:
            output_file_path = os.path.join(output_path, f"{document_name}.json")
            if os.path.exists(output_file_path):
                continue
            
            with open(os.path.join(self.data_path, document_name), "r") as f:
                data = json.load(f)

            qa_pairs = self.llm_service.complete(
                    system_prompt=self.QUESTION_SYSTEM_PROMPT,
                    user_prompt=f"Please create me {self.num_questions} sets of questions/answers based on the following tables: {data}",
                    providers=self.providers,
                    json_output=True,
                )
            
            with open(os.path.join(output_path, f"{document_name}.json"), "w") as f:
                json.dump(qa_pairs, f)