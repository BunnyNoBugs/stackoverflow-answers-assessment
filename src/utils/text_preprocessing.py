from bs4 import BeautifulSoup
from typing import Optional, Union, List


def clean_html(html_doc, preserve_tags: Optional[Union[str, List[str]]] = None):
    """Clean HTML but preserve selected tags."""
    if isinstance(preserve_tags, str):
        preserve_tags = [preserve_tags]
    elif preserve_tags is None:
        preserve_tags = []

    soup = BeautifulSoup(html_doc, 'html.parser')

    for tag in soup.find_all(True):
        if tag.name not in preserve_tags:
            tag.unwrap()

    return str(soup)


class Preprocessor:
    """
    Preprocess text of question-answer pairs from Dalip-like dataset.
    """

    def __init__(
            self,
            question_columns: Optional[Union[str, List[str]]] = 'default',
            answer_columns: Union[str, List[str]] = 'answer_body',
            preserve_html_tags: Optional[Union[str, List[str]]] = None
    ):
        if question_columns == 'default':
            question_columns = ['question_title', 'question_body']
        elif isinstance(question_columns, str):
            question_columns = [question_columns]
        elif question_columns is None:
            question_columns = []
        self.question_columns = question_columns

        if isinstance(answer_columns, str):
            answer_columns = [answer_columns]
        self.answer_columns = answer_columns

        self.preserve_html_tags = preserve_html_tags

    def __call__(self, examples):
        if isinstance(examples[self.answer_columns[0]], list):
            batch_size = len(examples[self.answer_columns[0]])
        else:
            examples = {col: [feature] for col, feature in examples.items()}
            batch_size = 1

        question_texts = []
        answer_texts = []

        for i in range(batch_size):
            question_texts.append(
                '\n'.join([clean_html(examples[col][i], self.preserve_html_tags) for col in self.question_columns]))
            answer_texts.append(
                '\n'.join([clean_html(examples[col][i], self.preserve_html_tags) for col in self.answer_columns]))

        examples['question_text'] = question_texts
        examples['answer_text'] = answer_texts

        return examples


if __name__ == '__main__':
    from datasets import load_from_disk
    from src.utils.config_management import CONFIG

    hf_dataset = load_from_disk(CONFIG['paths']['data']['dalip_hf_dataset'])

    preprocessor = Preprocessor()

    hf_dataset.map(preprocessor)
