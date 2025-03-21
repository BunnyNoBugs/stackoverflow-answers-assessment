from typing import Tuple
import pandas as pd
from collections import defaultdict


class DalipDataset:
    pass


class PredictionTransformer:
    def __init__(self, dataset_name='dalip'):
        # can be adapted to other datasets in the future
        self.dataset_name = dataset_name

    def transform_to_rankings(self, predictions, target_dataset_df: pd.DataFrame) -> Tuple[dict]:
        """
        Transform predictions to rankings.
        :param target_dataset_df:
        :param predictions:
        :return: Tuple (true_scores, predicted_scores)
        """
        true_scores = defaultdict(list)
        predicted_scores = defaultdict(list)

        for (_, row), prediction in zip(target_dataset_df.iterrows(), predictions):
            if self.dataset_name == 'dalip':
                question_id = row['ParentId']
                true_scores[question_id].append(row['Score'])
                predicted_scores[question_id].append(prediction)
            else:
                raise NotImplementedError

        return true_scores, predicted_scores
