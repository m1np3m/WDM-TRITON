from deepeval.test_case import MLLMTestCase, MLLMImage
from src.services.eval.rag_eval import RAGEval

class MultiModalEval(RAGEval):
    def __init__(self, questions: list[str], predictions: list[list[str]], ground_truths: list[list[str]], retrieval_context: list[list[str]]):
        super().__init__(questions, predictions, ground_truths, retrieval_context)
    
    def make_test_case(self) -> list[MLLMTestCase]:
        test_cases = []
        for question, prediction, ground_truth, retrieval_context in zip(self.questions, self.predictions, self.ground_truths, self.retrieval_context):
            test_case = MLLMTestCase(
                input=[question],
                expected_output=[
                    MLLMImage(url=pred) if (pred.endswith('.png') or pred.endswith('.jpg')) else pred
                    for pred in prediction
                ],
                actual_output=[
                    MLLMImage(url=gt) if (gt.endswith('.png') or gt.endswith('.jpg')) else gt
                    for gt in ground_truth
                ],
                retrieval_context=[
                    MLLMImage(url=context) if (context.endswith('.png') or context.endswith('.jpg')) else context
                    for context in retrieval_context
                ],
                context=[
                    MLLMImage(url=context) if (context.endswith('.png') or context.endswith('.jpg')) else context
                    for context in retrieval_context
                ]
            )
            test_cases.append(test_case)
        return test_cases        

