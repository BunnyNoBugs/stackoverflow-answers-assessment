from src.data_management import pairs_to_rankings
from typing import Literal, List, Union
from sklearn.metrics import ndcg_score, mean_absolute_error


def compute_ndcg(true_rankings, predicted_rankings, k):
    """Compute mean nDCG@k for predicted rankings."""
    ndcg_values = []
    for true_ranking, predicted_ranking in zip(true_rankings.values(), predicted_rankings.values()):
        ndcg_values.append(ndcg_score([true_ranking], [predicted_ranking], k=k))

    return sum(ndcg_values) / len(ndcg_values)


def compute_mae(targets, predictions):
    """Compute mean absolute error for predictions."""
    return mean_absolute_error(targets, predictions)


class RankingEvaluator:
    """Evaluate ranking metrics on question-answer pairs."""

    def __init__(self,
                 ndcg_k: Union[Union[int, Literal['all']], List[Union[int, Literal['all']]]] = 'all'
                 ):
        if isinstance(ndcg_k, list):
            self.ndcg_k = ndcg_k
        else:
            self.ndcg_k = [ndcg_k]

    def __call__(self, targets, predictions, group_ids):
        true_rankings, predicted_rankings = pairs_to_rankings(targets, predictions, group_ids)

        results = {}

        for ndcg_k in self.ndcg_k:
            if ndcg_k == 'all':
                ndcg_value = compute_ndcg(true_rankings, predicted_rankings, None)
            else:
                ndcg_value = compute_ndcg(true_rankings, predicted_rankings, ndcg_k)

            results[f'ndcg@{ndcg_k}'] = ndcg_value

        results['mae'] = compute_mae(targets, predictions)

        return results
