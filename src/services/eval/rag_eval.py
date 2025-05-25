import os
from deepeval import evaluate
from deepeval.metrics import MultimodalAnswerRelevancyMetric, MultimodalFaithfulnessMetric
from deepeval.metrics import MultimodalContextualPrecisionMetric, MultimodalContextualRecallMetric, MultimodalContextualRelevancyMetric
from deepeval.test_case import MLLMTestCase, MLLMImage, LLMTestCase
from typing import Optional, Union
from deepeval.models import MultimodalGeminiModel
from dotenv import load_dotenv

load_dotenv()

class RAGEval:
    def __init__(self, questions: list[str], predictions: list[list[str]], ground_truths: list[list[str]], retrieval_context: list[list[str]]):
        self.questions = questions
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.retrieval_context = retrieval_context

        self.model = MultimodalGeminiModel(
            model_name="gemini-2.5-flash-preview-05-20",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        self.answer_relevancy_metric = MultimodalAnswerRelevancyMetric(model=self.model)
        self.faithfulness_metric = MultimodalFaithfulnessMetric(model=self.model)
        self.contextual_precision_metric = MultimodalContextualPrecisionMetric(model=self.model)
        self.contextual_recall_metric = MultimodalContextualRecallMetric(model=self.model)
        self.contextual_relevancy_metric = MultimodalContextualRelevancyMetric(model=self.model)


    def evaluate(self, test_cases: list[Union[MLLMTestCase, LLMTestCase]]):
        results = []
        for test_case in test_cases:
            self.answer_relevancy_metric.measure(test_case)
            self.faithfulness_metric.measure(test_case)
            self.contextual_precision_metric.measure(test_case)
            self.contextual_recall_metric.measure(test_case)
            self.contextual_relevancy_metric.measure(test_case)

            results.append({
                "answer_relevancy": {"score": self.answer_relevancy_metric.score, "reason": self.answer_relevancy_metric.reason},
                "faithfulness": {"score": self.faithfulness_metric.score, "reason": self.faithfulness_metric.reason},
                "contextual_precision": {"score": self.contextual_precision_metric.score, "reason": self.contextual_precision_metric.reason},
                "contextual_recall": {"score": self.contextual_recall_metric.score, "reason": self.contextual_recall_metric.reason},
                "contextual_relevancy": {"score": self.contextual_relevancy_metric.score, "reason": self.contextual_relevancy_metric.reason}
            })
        return results

    def evaluate_retrieval(self):
        pass
    
    def make_test_case(self):
        pass