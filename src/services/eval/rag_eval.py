import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance, Faithfulness, ResponseRelevancy
from ragas import evaluate, EvaluationDataset

load_dotenv()

class RAGEval:
    def __init__(self, questions: list[str], predictions: list[list[str]], ground_truths: list[list[str]], retrieval_context: list[list[str]]):
        self.questions = questions
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.retrieval_context = retrieval_context

        self.model = GoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.evaluator_llm = LangchainLLMWrapper(self.model)

    def evaluate(self, test_cases: list):
        dataset = EvaluationDataset.from_list(test_cases)
        results = evaluate(dataset=dataset, llm=self.evaluator_llm, metrics=[Faithfulness(), ResponseRelevancy()])
        return results.scores

    def evaluate_retrieval(self):
        pass
    
    def make_test_case(self):
        pass