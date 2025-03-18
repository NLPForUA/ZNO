from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

from tools.logger import logging

DETAILED = "розгорнуту "
USING_OPTIONS = ' та використовуючи лише наведені нижче варіанти'
ONLY_LETTERS = ' У якості відповіді наведіть лише літеру, що відповідає правильному варіанту. Якщо правильних відповідей декілька, то перерахуйте їх через ";".'
ANSWER = 'Відповідь:'
TOPIC = 'ТЕМА'

def get_prompt():
    return 'Дайте {detailed}відповідь на завдання, починаючи з ключового слова "Відповідь:"{using_options}.{only_letters}\n\nЗавдання: {question}\n\n{options}\n\n{thoughts}'


def get_chat_prompt():
    return 'Дайте {detailed}відповідь на завдання, починаючи з ключового слова "Відповідь:"{using_options}.{only_letters}\n\nЗавдання: {question}\n\n{options}'


def format_prompt(row, cot=True, chat_like=False, with_topic=True):
    detailed = ''
    using_options = USING_OPTIONS
    only_letters = ONLY_LETTERS #' У відповіді перераховуйте через ";" тільки літери та цифри, що відповідають правильним варіантам.'
    options = ""

    answer_options = row['answers']
    
    if not answer_options:
        detailed = DETAILED
        using_options = ""
        only_letters = ""
    else:
        options = "Варіанти відповіді:\n"
        options += "\n".join(f"{a['answer']} - {a['text']}" for a in answer_options)
    
    if cot:
        detailed = DETAILED
        only_letters = ""
        if with_topic:
            thoughts = f"{ANSWER}\n" + row['comment'].replace('Коментар\n', '').strip()
        else:
            if row['comment'] != "" and 'Зразок написання власного висловлення' not in row['comment']:
                lines = row['comment'].split('\n')
                if lines[1].strip().startswith(f"{TOPIC}:") == True or lines[1].strip().startswith(f"{TOPIC}.") == True or lines[1].strip().startswith(f"{TOPIC} :") == True:
                    thoughts = '\n'.join(lines[2:]).strip()
                else:
                    thoughts = row['comment'].strip()
            else:
                thoughts = row['comment'].strip()
            thoughts = f"{ANSWER}\n" + thoughts

    else:
        thoughts = ANSWER + " " + (row['correct_answer'][0].strip() if len(row['correct_answer']) == 1 else ';'.join(row['correct_answer']).strip())


    if chat_like:
        return (get_chat_prompt().format(
        question=row["question"],
        options=options,
        using_options=using_options,
        only_letters=only_letters,
        detailed=detailed,
    ), thoughts)

    return get_prompt().format(
        question=row["question"],
        options=options,
        using_options=using_options,
        only_letters=only_letters,
        detailed=detailed,
        thoughts=thoughts
    )


def apply_template(tokenizer: AutoTokenizer, dataset: List[Any], debug: bool = True, replace_map: Optional[Dict[str, str]] = None, date_string=None) -> Dict[str, List[str]]:
    logging.info(f"Templating dataset with {len(dataset)} rows and replace_map={replace_map}\n")
    texts = []
    warned = False
    for row in dataset:
        messages = [
            #{"role": "system", "content": "You are a highly intelligent assistant taking the graduation exam in a Ukrainian school"}, 
            {"role": "user", "content": row[0]}, 
            {"role": "assistant", "content": row[1]}
        ]
        
        additional_args = {}
        if date_string is not None:
            additional_args["date_string"] = date_string

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **additional_args)
        if replace_map is not None and len(replace_map) > 0:
            for k, v in replace_map.items():
                text = text.replace(k, v)
        if "llama" in str(tokenizer.name_or_path).lower():
            if not warned and text.startswith("<|begin_of_text|>"):
                logging.info("Warning: templated text starts with <|begin_of_text|> token, while some tokenizers add this token automatically. Make sure it's not added twice.")
                warned = True
        texts.append(text)
    
    if debug:
        for row in texts[:5]:
            logging.info(row)
            logging.info("\n---------------------------------------------------\n")
    
    return {"text": texts}