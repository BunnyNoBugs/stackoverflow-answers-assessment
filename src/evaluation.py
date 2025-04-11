from src.data_management import pairs_to_rankings
from typing import Literal, List, Union
from sklearn.metrics import mean_absolute_error
import numpy as np


def compute_custom_ndcg(true_rankings, predicted_scores, k=None, gain_func='linear', discount_func='logarithmic'):
    """
    Compute mean nDCG@k for predicted rankings with customizable gain and discount functions.
    Gain function options:
        - exponential
        - linear
    Discount function options:
        - logarithmic
        - zipfian
    """

    def compute_gain(relevance, func):
        if func == 'exponential':
            return 2 ** relevance - 1
        elif func == 'linear':
            return relevance

    def compute_discount(rank, func):
        if func == 'logarithmic':
            return 1 / np.log2(rank + 1)
        if func == 'zipfian':
            return 1 / rank

    def compute_dcg(relevances, gain_func, discount_func):
        return sum(compute_gain(rel, gain_func) * compute_discount(rank + 1, discount_func)
                   for rank, rel in enumerate(relevances))

    ndcg_values = []
    for true_relevances, prediction in zip(true_rankings.values(), predicted_scores.values()):
        true_relevances = np.array(true_relevances, dtype=np.float64)
        prediction = np.array(prediction, dtype=np.float64)

        if k:
            predicted_ranking = true_relevances[prediction.argsort()][::-1][:k]
            true_ranking = np.sort(true_relevances)[::-1][:k]
        else:
            predicted_ranking = true_relevances[prediction.argsort()][::-1]
            true_ranking = np.sort(true_relevances)[::-1]

        dcg_value = compute_dcg(predicted_ranking, gain_func, discount_func)
        idcg_value = compute_dcg(true_ranking, gain_func, discount_func)

        ndcg_value = dcg_value / idcg_value
        ndcg_values.append(ndcg_value)

    return sum(ndcg_values) / len(ndcg_values)


def compute_mae(targets, predictions):
    """Compute mean absolute error for predictions."""
    return mean_absolute_error(targets, predictions)


def compute_hit_rate_at_1(true_rankings, predicted_rankings):
    """
    Compute mean hit rate @ 1 for predicted rankings.
    This metric returns 1 if any of the items with maximum relevance has been ranked first, and 0 otherwise.
    """
    hit_rates = []
    for true_ranking, predicted_ranking in zip(true_rankings.values(), predicted_rankings.values()):
        true_ranking = np.array(true_ranking)
        predicted_ranking = np.array(predicted_ranking)

        max_relevance = true_ranking.max()

        hit = int(true_ranking[np.argmax(predicted_ranking)] == max_relevance)
        hit_rates.append(hit)

    return sum(hit_rates) / len(hit_rates)


class RankingEvaluator:
    """Evaluate ranking metrics on question-answer pairs."""

    # todo: refactor predicted_rankings to predicted_scores_dict
    # true_rankings -> true_relevances_dict

    def __init__(self,
                 ndcg_k: Union[Union[int, Literal['all']], List[Union[int, Literal['all']]]] = 'all',
                 ndcg_gain_func='linear',
                 ndcg_discount_func='logarithmic'
                 ):
        if isinstance(ndcg_k, list):
            self.ndcg_k = ndcg_k
        else:
            self.ndcg_k = [ndcg_k]

        self.ndcg_gain_func = ndcg_gain_func
        self.ndcg_discount_func = ndcg_discount_func

    def __call__(self, targets, predictions, group_ids):
        true_rankings, predicted_rankings = pairs_to_rankings(targets, predictions, group_ids)

        results = {}

        for ndcg_k in self.ndcg_k:
            if ndcg_k == 'all':
                ndcg_value = compute_custom_ndcg(true_rankings, predicted_rankings,
                                                 None, self.ndcg_gain_func, self.ndcg_discount_func)
            else:
                ndcg_value = compute_custom_ndcg(true_rankings, predicted_rankings,
                                                 ndcg_k, self.ndcg_gain_func, self.ndcg_discount_func)

            results[f'g.{self.ndcg_gain_func}_d.{self.ndcg_discount_func}_ndcg@{ndcg_k}'] = ndcg_value

        results['mae'] = compute_mae(targets, predictions)

        results['hit_rate@1'] = compute_hit_rate_at_1(true_rankings, predicted_rankings)

        return results
