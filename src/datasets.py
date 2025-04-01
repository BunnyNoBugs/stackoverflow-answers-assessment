from typing import Tuple
import pandas as pd
from collections import defaultdict
from datasets import Dataset, Value, DatasetDict
from sklearn.model_selection import train_test_split


def dalip_normalize_answer_scores(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize answer scores in a Dalip-like dataset to avoid negative values in nDCG computation.
    :param dataset_df:
    :return:
    """
    answers_mask = dataset_df['PostTypeId'] == 2

    dataset_df['NormalizedScore'] = pd.NA

    min_answer_score = dataset_df.loc[answers_mask, 'Score'].min()

    dataset_df.loc[answers_mask, 'NormalizedScore'] = dataset_df.loc[answers_mask, 'Score'] - min_answer_score

    return dataset_df


def dalip_dataset_create_pairs(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a Dalip-like dataset from a collections of posts to question-answer pairs.
    :param dataset_df:
    :return:
    """
    question_features = [
        'Id', 'CreationDate', 'Score', 'ViewCount', 'Body', 'LastEditDate', 'LastActivityDate', 'Title', 'Tags',
        'AnswerCount',
        'CommentCount', 'FavoriteCount', 'ClosedDate', 'CommunityOwnedDate', 'AcceptedAnswerId']
    questions_df = dataset_df[dataset_df['PostTypeId'] == 1][question_features].rename(
        columns={
            'Id': 'question_id',
            'CreationDate': 'question_creation_date',
            'Score': 'question_score',
            'ViewCount': 'question_view_count',
            'Body': 'question_body',
            'LastEditDate': 'question_last_edit_date',
            'LastActivityDate': 'question_last_activity_date',
            'Title': 'question_title',
            'Tags': 'question_tags',
            'AnswerCount': 'question_answer_count',
            'CommentCount': 'question_comment_count',
            'FavoriteCount': 'question_favorite_count',
            'ClosedDate': 'question_closed_date',
            'CommunityOwnedDate': 'question_community_owned_date',
            'AcceptedAnswerId': 'accepted_answer_id'
        })

    answer_features = ['Id', 'ParentId', 'CreationDate', 'Score', 'NormalizedScore', 'Body', 'LastEditDate',
                       'LastActivityDate', 'CommentCount', 'CommunityOwnedDate']
    answers_df = dataset_df[dataset_df['PostTypeId'] == 2][answer_features].rename(
        columns={
            'Id': 'answer_id',
            'ParentId': 'question_id',
            'CreationDate': 'answer_creation_date',
            'Score': 'answer_score',
            'NormalizedScore': 'answer_normalized_score',
            'Body': 'answer_body',
            'LastEditDate': 'answer_last_edit_date',
            'LastActivityDate': 'answer_last_activity_date',
            'CommentCount': 'answer_comment_count',
            'CommunityOwnedDate': 'answer_community_owned_date'
        })

    qa_pairs_df = pd.merge(answers_df, questions_df, on='question_id', how='inner')
    qa_pairs_df['answer_accepted'] = qa_pairs_df['answer_id'] == qa_pairs_df['accepted_answer_id']
    qa_pairs_df = qa_pairs_df.drop(columns=['accepted_answer_id'])

    return qa_pairs_df


def dalip_dataset_to_huggingface(dataset_df: pd.DataFrame, test_size: float = 0.2,
                                 random_state: int = 42) -> DatasetDict:
    """
    Convert a Dalip-like dataset to a HuggingFace Dataset of question-answer pairs.
    Additional operations:
    - normalize answer scores
    - perform a train-test split.
    :param test_size:
    :param random_state:
    :param dataset_df:
    :return:
    """
    dataset_df = dalip_normalize_answer_scores(dataset_df)

    question_ids = dataset_df[dataset_df['PostTypeId'] == 1]['Id'].unique()
    train_question_ids, test_question_ids = train_test_split(question_ids, test_size=test_size,
                                                             random_state=random_state)
    train_df = dataset_df[dataset_df['ParentId'].isin(train_question_ids) | dataset_df['Id'].isin(train_question_ids)]
    test_df = dataset_df[dataset_df['ParentId'].isin(test_question_ids) | dataset_df['Id'].isin(test_question_ids)]

    train_qa_pairs_df = dalip_dataset_create_pairs(train_df)
    test_qa_pairs_df = dalip_dataset_create_pairs(test_df)

    train_hf_dataset = Dataset.from_pandas(train_qa_pairs_df, preserve_index=False).cast_column('question_id',
                                                                                                Value('int64'))
    test_hf_dataset = Dataset.from_pandas(test_qa_pairs_df, preserve_index=False).cast_column('question_id',
                                                                                              Value('int64'))

    hf_dataset = DatasetDict({'train': train_hf_dataset, 'test': test_hf_dataset})

    return hf_dataset


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
