from deepeval.test_case import MLLMTestCase, MLLMImage
from src.services.eval.rag_eval import RAGEval

def calculate_precision(predictions: list[str], ground_truths: str) -> float:
    """
    Calculate precision score between predictions and ground truths.
    Precision = (True Positives) / (True Positives + False Positives)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    true_positives = sum(1 for pred in predictions if pred in ground_truths)
    return true_positives / len(predictions) if predictions else 0.0

def calculate_recall(predictions: list[str], ground_truths: str) -> float:
    """
    Calculate recall score between predictions and ground truths.
    Recall = (True Positives) / (True Positives + False Negatives)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    true_positives = sum(1 for pred in predictions if pred in ground_truths)
    return true_positives / len(ground_truths) if ground_truths else 0.0

def calculate_f1_score(predictions: list[str], ground_truths: str) -> float:
    """
    Calculate F1 score between predictions and ground truths.
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = calculate_precision(predictions, ground_truths)
    recall = calculate_recall(predictions, ground_truths)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

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
                    MLLMImage(url=context) if (str(context).endswith('.png') or str(context).endswith('.jpg')) else context
                    for context in retrieval_context
                ],
                context=[
                    MLLMImage(url=context) if (str(context).endswith('.png') or str(context).endswith('.jpg')) else context
                    for context in retrieval_context
                ]
            )
            test_cases.append(test_case)
        return test_cases        

    def evaluate_retrieval(self):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            precision_scores.append(calculate_precision(pred, gt[0]))
            recall_scores.append(calculate_recall(pred, gt[0]))
            f1_scores.append(calculate_f1_score(pred, gt[0]))
        
        return {
            "precision": sum(precision_scores) / len(precision_scores),
            "recall": sum(recall_scores) / len(recall_scores),
            "f1_score": sum(f1_scores) / len(f1_scores)
        }