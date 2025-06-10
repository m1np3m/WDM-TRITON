from src.services.eval.rag_eval import RAGEval

def calculate_precision_at_k(predictions: list[list[str]], ground_truths: list[str], k: int = 1) -> float:
    """
    Calculate precision score at K between predictions and ground truths.
    Precision = (True Positives) / (True Positives + False Positives)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # true_positives = sum(1 for pred in predictions[:k] if any(_pred in ground_truths for _pred in pred))
    true_positives = sum(1 for pred in predictions[:k] if set(pred).issubset(set(ground_truths)) or set(ground_truths).issubset(set(pred)))

    return true_positives / len(predictions[:k]) if predictions else 0.0

def calculate_recall_at_k(predictions: list[list[str]], ground_truths: list[str], k: int = 1) -> float:
    """
    Calculate recall score at K between predictions and ground truths.
    Recall = (True Positives) / (True Positives + False Negatives)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # true_positives = sum(1 for pred in predictions if any(_pred in ground_truths for _pred in pred))
    true_positives = sum(1 for pred in predictions if set(pred).issubset(set(ground_truths)) or set(ground_truths).issubset(set(pred)))

    # return true_positives / len(ground_truths) if ground_truths else 0.0
    return min(true_positives / 1, 1.0) if ground_truths else 0.0

def calculate_f1_at_k(predictions: list[list[str]], ground_truths: list[str], k: int = 1) -> float:
    """
    Calculate F1 score at K between predictions and ground truths.
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = calculate_precision_at_k(predictions, ground_truths, k)
    recall = calculate_recall_at_k(predictions, ground_truths, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_ap_at_k(predictions: list[list[str]], ground_truths: list[str], k: int = 1) -> float:
    """
    Calculate Average Precision at K (AP@K) between predictions and ground truths.
    AP@K = (1/K) * Σ(precision@i * rel(i)) where i is the position and rel(i) is 1 if the item is relevant
    
    Args:
        predictions: List of predicted items, where each item is a list of strings
        ground_truths: List of ground truth items
        k: Number of top results to consider
    
    Returns:
        float: Average Precision at K score
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # Initialize variables
    ap = 0.0
    num_relevant = 0
    
    # Calculate AP@K
    for i in range(min(k, len(predictions))):
        if set(predictions[i]).issubset(set(ground_truths)) or set(ground_truths).issubset(set(predictions[i])):
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            ap += precision_at_i
    
    # Normalize by the number of relevant items or k, whichever is smaller
    return ap / min(k, len(ground_truths)) if ground_truths else 0.0

def calculate_mrr_at_k(predictions: list[list[str]], ground_truths: list[str], k: int = 1) -> float:
    """
    Calculate Mean Reciprocal Rank at K (MRR@K) between predictions and ground truths.
    MRR@K = 1/rank of first relevant item, where rank is the position of the first relevant item in top K results
    
    Args:
        predictions: List of predicted items, where each item is a list of strings
        ground_truths: List of ground truth items
        k: Number of top results to consider
    
    Returns:
        float: MRR@K score (0 if no relevant items found in top K)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # Find the rank of first relevant item
    for i in range(min(k, len(predictions))):
        if set(predictions[i]).issubset(set(ground_truths)) or set(ground_truths).issubset(set(predictions[i])):
            return 1.0 / (i + 1)
    
    return 0.0

class MultiModalEval(RAGEval):
    def __init__(self, questions: list[str], predictions: list[list[list[str]]], ground_truths: list[list[str]], retrieval_context: list[list[str]] = None):
        super().__init__(questions, predictions, ground_truths, retrieval_context)
    
    def make_test_case(self) -> list:
        test_cases = []
        for question, prediction, ground_truth, retrieval_context in zip(self.questions, self.predictions[0], self.ground_truths, self.retrieval_context):
            test_case = {
                "user_input": question,
                "retrieved_contexts": [context for context in retrieval_context],
                "response": prediction[0],
                "reference": ground_truth[0]
            }
            test_cases.append(test_case)

        return test_cases

    def evaluate_retrieval(self, k: int = 1):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ap_scores = []
        mrr_scores = []
        results = []
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred = list(set(tuple(x) for x in pred))
            pred = [list(x) for x in pred]

            precision = calculate_precision_at_k(pred, gt, k)
            recall = calculate_recall_at_k(pred, gt, k)
            f1 = calculate_f1_at_k(pred, gt, k)
            ap = calculate_ap_at_k(pred, gt, k)
            mrr = calculate_mrr_at_k(pred, gt, k)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ap_scores.append(ap)
            mrr_scores.append(mrr)
            
            results.append({
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "ap_score": ap,
                "mrr_score": mrr
            })
        
        results.append({
            "precision": sum(precision_scores) / len(precision_scores),
            "recall": sum(recall_scores) / len(recall_scores),
            "f1_score": sum(f1_scores) / len(f1_scores),
            "ap_score": sum(ap_scores) / len(ap_scores),
            "mrr_score": sum(mrr_scores) / len(mrr_scores)
        })
        
        return results
