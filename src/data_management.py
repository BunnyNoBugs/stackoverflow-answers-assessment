from typing import Tuple, Optional, List, Union, Literal
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import random
from tqdm import tqdm
from datasets import Dataset, Value, DatasetDict
from sklearn.model_selection import train_test_split


def dalip_normalize_answer_scores(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to answer scores in a Dalip-like dataset.
    Transformations applied:
    - Min-shifting to avoid negative values in nDCG computation
    - Log transformation (ln (x + 1)) to reduce skewness.
    :param dataset_df:
    :return:
    """
    answers_mask = dataset_df['PostTypeId'] == 2

    dataset_df['NormalizedScore'] = pd.NA
    min_answer_score = dataset_df.loc[answers_mask, 'Score'].min()
    dataset_df.loc[answers_mask, 'NormalizedScore'] = dataset_df.loc[answers_mask, 'Score'] - min_answer_score

    dataset_df['LogNormalizedScore'] = pd.NA
    dataset_df.loc[answers_mask, 'LogNormalizedScore'] = np.log1p(
        dataset_df.loc[answers_mask, 'NormalizedScore'].astype(int))

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

    answer_features = ['Id', 'ParentId', 'CreationDate', 'Score', 'NormalizedScore', 'LogNormalizedScore',
                       'Body', 'LastEditDate', 'LastActivityDate', 'CommentCount', 'CommunityOwnedDate']
    answers_df = dataset_df[dataset_df['PostTypeId'] == 2][answer_features].rename(
        columns={
            'Id': 'answer_id',
            'ParentId': 'question_id',
            'CreationDate': 'answer_creation_date',
            'Score': 'answer_score',
            'NormalizedScore': 'answer_normalized_score',
            'LogNormalizedScore': 'answer_log_normalized_score',
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


def dalip_dataset_to_huggingface(dataset_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                                 test_question_ids: Optional[List[int]] = None) -> DatasetDict:
    """
    Convert a Dalip-like dataset to a HuggingFace Dataset of question-answer pairs.
    Additional operations:
    - normalize answer scores
    - perform a train-test split.
    :param test_size:
    :param random_state:
    :param dataset_df:
    :param test_question_ids:
    :return:
    """
    dataset_df = dalip_normalize_answer_scores(dataset_df)

    question_ids = pd.Series(dataset_df[dataset_df['PostTypeId'] == 1]['Id'].unique())

    if test_question_ids is None:
        train_question_ids, test_question_ids = train_test_split(question_ids, test_size=test_size,
                                                                 random_state=random_state)
    else:
        train_question_ids = question_ids[~question_ids.isin(test_question_ids)]

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


def pairs_to_rankings(targets, predictions, group_ids) -> Tuple[dict, dict]:
    """
    Transform pair relevance predictions to rankings.
    :param targets:
    :param predictions:
    :param group_ids:
    :return:
    """
    true_scores = defaultdict(list)
    predicted_scores = defaultdict(list)

    for prediction, target, group_id in zip(predictions, targets, group_ids):
        true_scores[group_id].append(target)
        predicted_scores[group_id].append(prediction)

    return true_scores, predicted_scores


def create_pairs_dataset_df(dataset_df,
                            pairs_sampling_strategy: Union[Literal['mean'], Literal['topk']] = 'mean',
                            n: Union[int, Literal['all']] = 'all',
                            TARGET_COL='answer_normalized_score'
                            ) -> pd.DataFrame:
    """
    Create a dataset of answer pairs out of a dataset of question-answer pairs.
    ``pairs_sampling_strategy`` and ``n`` options follow those of
    ``lambdarank_pair_method`` and ``lambdarank_num_pair_per_sample`` in xgboost.
    :param dataset_df:
    :param pairs_sampling_strategy:
    :param n:
    :param TARGET_COL:
    :return:
    """

    def create_pairs(group, pairs_sampling_strategy, n):
        group = group.sort_values(TARGET_COL, ascending=False)

        all_pairs_idxs = list(combinations(group.index, 2))
        pairs_idxs = []

        if n == 'all':
            for pair_idx in all_pairs_idxs:
                if group.loc[pair_idx[0]][TARGET_COL] != group.loc[pair_idx[1]][TARGET_COL]:
                    pairs_idxs.append(pair_idx)

        else:
            if pairs_sampling_strategy == 'mean':
                random.shuffle(all_pairs_idxs)
                for pair_idx in all_pairs_idxs:
                    if group.loc[pair_idx[0]][TARGET_COL] != group.loc[pair_idx[1]][TARGET_COL]:
                        pairs_idxs.append(pair_idx)
                    if len(pairs_idxs) == n:
                        break

            elif pairs_sampling_strategy == 'topk':
                for curr_k in range(min(n, len(group))):
                    anchor_idx = group.index[curr_k]
                    for idx in group.index[curr_k:]:
                        if group.loc[anchor_idx][TARGET_COL] != group.loc[idx][TARGET_COL]:
                            pairs_idxs.append((anchor_idx, idx))

        pairs = []
        for pair_idx in pairs_idxs:
            pair = {
                'question_id': group.loc[pair_idx[0]]['question_id'],
                'answer_1_id': group.loc[pair_idx[0]]['answer_id'],
                'answer_2_id': group.loc[pair_idx[1]]['answer_id'],
                'question_text': group.loc[pair_idx[0]]['question_text'],
                'answer_1_text': group.loc[pair_idx[0]]['answer_text'],
                'answer_2_text': group.loc[pair_idx[1]]['answer_text'],
                f'answer_1_{TARGET_COL}': group.loc[pair_idx[0]][TARGET_COL],
                f'answer_2_{TARGET_COL}': group.loc[pair_idx[1]][TARGET_COL],
            }
            if group.loc[pair_idx[0]][TARGET_COL] > group.loc[pair_idx[1]][TARGET_COL]:
                pair['label'] = 1
            else:
                pair['label'] = -1

            pairs.append(pair)

        return pairs

    groups = dataset_df.groupby('question_id')

    pairs_dataset_df = []
    for name, group in tqdm(groups):
        group_pairs = create_pairs(group, pairs_sampling_strategy, n)
        pairs_dataset_df.extend(group_pairs)
    pairs_dataset_df = pd.DataFrame(pairs_dataset_df)

    return pairs_dataset_df
