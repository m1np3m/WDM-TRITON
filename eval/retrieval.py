from typing import List


class RetrievaEvaluation:
    """
    A class for evaluating retrieval performance of RAG.
    It provides static methods to calculate metrics.
    Each method takes a list of ground truth items that each of them is the relevant table used for answering the question, and a list of predictions of relevant tables for questions.
    """

    @staticmethod
    def calculate_precision_at_k(
        predictions: List[List[str]], ground_truths: List[str], top_k: int = 1
    ) -> float:
        """
        Precision@k = (Number of relevant items in the top k results) / k
        """
        if not predictions or not ground_truths:
            raise ValueError("Predictions and ground truths cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        top_k = min(top_k, len(predictions))

        precisions = sum(
            1 / top_k
            for i, prediction in enumerate(predictions)
            if ground_truths[i] in prediction[:top_k]
        )

        return sum(precisions) / len(ground_truths)

    @staticmethod
    def calculate_recall_at_k(
        predictions: List[List[str]], ground_truths: List[str], top_k: int = 1
    ) -> float:
        """
        Recall@k = (Number of relevant items in the top k results) / (Total number of relevant items)
        """
        if not predictions or not ground_truths:
            raise ValueError("Predictions and ground truths cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        top_k = min(top_k, len(predictions))

        recals = sum(
            1
            for i, prediction in enumerate(predictions)
            if ground_truths[i] in prediction[:top_k]
        )

        return recals / len(ground_truths)

    @staticmethod
    def calculate_f1_score(
        predictions: List[List[str]], ground_truths: List[str], top_k: int = 1
    ) -> float:
        """
        F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)
        """
        if not predictions or not ground_truths:
            raise ValueError("Predictions and ground truths cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        top_k = min(top_k, len(predictions))

        precision = RetrievaEvaluation.calculate_precision_at_k(
            predictions, ground_truths, top_k
        )
        recall = RetrievaEvaluation.calculate_recall_at_k(
            predictions, ground_truths, top_k
        )

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_ap_at_k(
        predictions: List[List[str]], ground_truths: List[str], top_k: int = 1
    ) -> float:
        """
        AP@k = (1 / R) * Σ (Precision@i * rel_i) for i = 1 to k,
        where:
        - R is the number of relevant items in the ground truth
        - rel_i is 1 if the item at position i is relevant, otherwise 0
        """
        if not predictions or not ground_truths:
            raise ValueError("Predictions and ground truths cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        top_k = min(top_k, len(predictions))

        sum_ap = 0.0

        for i, prediction in enumerate(predictions):
            if ground_truths[i] in prediction[:top_k]:
                sum_ap += 1 / (prediction.index(ground_truths[i]) + 1)

        return sum_ap / len(ground_truths)

    @staticmethod
    def calculate_mrr_at_k(
        predictions: List[List[str]], ground_truths: List[str], top_k: int = 1
    ) -> float:
        """
        MRR = (1 / N) * Σ (1 / rank_i),
        where:
        - N is the number of queries
        - rank_i is the rank of the first relevant item for the i-th query
        """
        if not predictions or not ground_truths:
            raise ValueError("Predictions and ground truths cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")

        top_k = min(top_k, len(predictions))

        mrr = 0.0

        for i, prediction in enumerate(predictions):
            if ground_truths[i] in prediction[:top_k]:
                mrr += 1 / (prediction.index(ground_truths[i]) + 1)

        return mrr / len(ground_truths)

    @staticmethod
    def evaluate_retrieval(
        predictions: List[List[str]], ground_truth: List[str], top_k: int = 1
    ) -> dict:
        """Evaluate retrieval performance metrics."""
        return {
            "precision_at_k": RetrievaEvaluation.calculate_precision_at_k(
                predictions, ground_truth, top_k
            ),
            "recall_at_k": RetrievaEvaluation.calculate_recall_at_k(
                predictions, ground_truth, top_k
            ),
            "f1_score": RetrievaEvaluation.calculate_f1_score(
                predictions, ground_truth, top_k
            ),
            "ap_at_k": RetrievaEvaluation.calculate_ap_at_k(
                predictions, ground_truth, top_k
            ),
            "mrr_at_k": RetrievaEvaluation.calculate_mrr_at_k(
                predictions, ground_truth, top_k
            ),
        }
