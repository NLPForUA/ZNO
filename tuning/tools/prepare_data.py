from collections import defaultdict
import hashlib
import json

from tools.logger import logging
from tools.prompt import ANSWER, DETAILED, TOPIC, format_prompt

def hash_dict(d: dict) -> str:
    d_json = json.dumps(d, sort_keys = True).encode("utf-8")
    return hashlib.md5(d_json).hexdigest()

def hash_dataset(dataset) -> str:
    return hash_dict(dataset)

def detect_duplicates(dataset, splits=["train", "val", "test"]):
    comments_set = defaultdict(set)
    answers_set = defaultdict(set)
    dups_stats = defaultdict(int)
    id_splits = defaultdict(set)

    skip_answers = set(
        ["1", "2", "3", "4", "іменник", "прикметник", "дієслово", "займенник", "числівник", "прислівник", "сполучник", "прийменник", "частка", "першому", "другому", "третьому", "четвертому", "п'ятому", "шостому"] 
    )

    for split in splits:
        for row in dataset[split]:
            id_splits[split].add(row["test_id"])
            str_comment = row["comment"].lower()

            if str_comment in comments_set[split]:
                dups_stats[f"{split}_comment_in_set"] += 1
                logging.info(f"Duplicate comment in {split} split: {row}")
            
            if len(str_comment) > 0:
                comments_set[split].add(str_comment)

            if not any([ans['text'].strip() in skip_answers for ans in row['answers']]):
                str_answer = '###'.join(sorted(row['text'] for row in row['answers'])).lower()
                if str_answer and str_answer in answers_set[split]:
                    dups_stats[f"{split}_answer_in_set"] += 1
                    logging.info(f"Duplicate answer in {split} split: {row}")
                answers_set[split].add(str_answer)

            for _split in splits:
                if _split == split:
                    continue
                if row["test_id"] in id_splits[_split]:
                    dups_stats[f"{split}_id_in_{_split}"] += 1
                    logging.info(f"\nDuplicate id in {split} and {_split} splits: {row}")
                if str_comment in comments_set[_split]:
                    dups_stats[f"{split}_comment_in_{_split}"] += 1
                    logging.info(f"\nDuplicate comment in {split} and {_split} splits: {row}")
                if str_answer in answers_set[_split]:
                    dups_stats[f"{split}_answer_in_{_split}"] += 1
                    logging.info(f"\nDuplicate answer in {split} and {_split} splits: {row}")
    return dups_stats

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as fr:
        dataset = json.load(fr)

    logging.info(dataset.keys(), {split: len(x) for split, x in dataset.items()})
    return dataset

def craft_input(dataset, is_cot: bool, debug: bool = True, with_topic: bool = True, strict: bool = True):
    rows_without_answer = 0
    rows_without_detailed_answer = 0
    rows_without_topic = 0
    train_input = []
    for row in dataset:
        train_input.append(format_prompt(row, chat_like=True, cot=is_cot, with_topic=with_topic))
        assert len(train_input[-1]) == 2, f"Row does not contain 2 elements\n{train_input[-1]}"
        rows_without_answer += len(train_input[-1][1].strip().split(' ')) < 2
        rows_without_detailed_answer += len(train_input[-1][1].strip().split(' ')) < 10
        rows_without_topic += TOPIC not in train_input[-1][1]
        if strict:
            check_row(train_input[-1], is_cot=is_cot, with_topic=with_topic)

    logging.info("Stats\n")
    logging.info(f"total rows: {len(train_input)}\nrows without answer: {rows_without_answer}\nrows without detailed answer: {rows_without_detailed_answer}\nrows without topic: {rows_without_topic}")
    if debug:
        for row in train_input[:5]:
            if isinstance(row, tuple):
                logging.info(row[0])
                logging.info('\n--Answer--\n')
                logging.info(row[1])
                logging.info('\n-------------------\n')
                continue
        logging.info(row)
        logging.info('\n\n')

    return train_input


def check_row(row: str, is_cot: bool, with_topic: bool) -> bool:
    assert len(row) == 2, f"Row does not contain 2 elements\n{row}"
    assert ANSWER in row[-1], f"Row does not contain '{ANSWER}'\n{row}"
    
    if with_topic and is_cot and TOPIC not in row[-1]:
        logging.warning(f"\n\nRow does not contain '{TOPIC}'\n{row}\n\n")
    elif (not with_topic or not is_cot) and TOPIC in row[-1]:
        logging.warning(f"\n\nRow contains '{TOPIC}'\n{row}\n\n")
    
    if is_cot and DETAILED not in row[0]:
        logging.warning(f"\n\nCoT Row does not contain '{DETAILED}'\n{row}\n\n")
    if not is_cot and DETAILED in row[0]:
        logging.warning(f"\n\nNo CoT Row contains '{DETAILED}'\n{row}\n\n")
    if not is_cot and len(row[-1].split(' ')) >= 10:
        logging.warning(f"\n\nLetter only row contains more than 10 words\n{row}\n\n")