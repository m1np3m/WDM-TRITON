import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    MultiModalFaithfulness,
    MultiModalRelevance,
    Faithfulness,
    ResponseRelevancy,
)
from ragas import evaluate, EvaluationDataset

from typing import List, Dict, Any


class RagEvaluation:

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = None):
        self.model = GoogleGenerativeAI(
            model=model,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
        )

        self.evaluator = LangchainLLMWrapper(self.model)

    def _prepare_evaluation_dataset(
        self,
        queries: List[str],
        reference_responses: List[str],
        responses: List[str],
        relevant_docs: List[str],
    ) -> EvaluationDataset:
        num_queries = len(queries)

        if (
            num_queries == 0
            or len(reference_responses) != num_queries
            or len(responses) != num_queries
            or len(relevant_docs) != num_queries
        ):
            raise ValueError("All input lists must have the same non-zero length.")

        dataset = []
        for i in range(num_queries):
            dataset.append(
                {
                    "user_input": queries[i],
                    "retrieved_contexts": relevant_docs[i],
                    "response": responses[i],
                    "reference": reference_responses[i],
                }
            )
        dataset = EvaluationDataset.from_list(dataset)

        return dataset

    def evaluate_rag(
        self,
        queries: List[str],
        reference_responses: List[str],
        responses: List[str],
        relevant_docs: List[str],
    ) -> Dict[str, float]:
        dataset = self._prepare_evaluation_dataset(
            queries, reference_responses, responses, relevant_docs
        )

        results = evaluate(
            dataset=dataset,
            llm=self.evaluator_llm,
            metrics=[Faithfulness(), ResponseRelevancy()],
        )

        return results
