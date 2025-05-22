# ZNO-Eval: Exam questions and answers in Ukrainian

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/NLPForUA/ZNO/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/NLPForUA/ZNO/blob/main/DATA_LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/NLPForUA/ZNO)
![GitHub Repo stars](https://img.shields.io/github/stars/NLPForUA/ZNO?style=social)
[![arXiv](https://img.shields.io/badge/arXiv-2501.06715-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2501.06715)

This repository contains structured test tasks and model tuning scripts for multiple subjects from ZNO - Ukrainian External Independent Evaluation (Зовнішнє Незалежне Оцінювання) and NMT - National Multi-Subject Test (Національний Мультипредметний Тест).

Usage and License Notices: The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The code is Apache 2.0 licensed.

## Updates:
- 2025-05-18: The exam dataset `dumy-zno-ukrainian-math-history-geo-r1-o1`, featuring questions, topics, answers, graded predictions, and reasoning from DeepSeek R1 and OpenAI o1, has been [published on Hugging Face](https://huggingface.co/datasets/NLPForUA/dumy-zno-ukrainian-math-history-geo-r1-o1).
- 2025-03-19: Preprint `Empowering Smaller Models: Tuning LLaMA and Gemma with Chain-of-Thought for Ukrainian Exam Tasks` with PEFT Chain-of-Thought tuning results submitted to [arXiv](https://arxiv.org/abs/2503.13988)
- 2025-03-18: [PEFT Chain-of-Thought tuning code and notebooks](tuning/), [Gemma and LLaMA models](https://huggingface.co/NLPForUA)
- 2025-01-19: Fixed question and answer options for Ukrainian language ZNO tasks (fixed 109 tasks missing hightlight of words or letters, using images as answer options or question descriptions)
- 2025-01-12: Paper `ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian` submitted to [arXiv](https://arxiv.org/abs/2501.06715)
- 2024-09-26: Paper `ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian` with baseline zero-shot presented at [X International conference "Informatics. Culture. Technology."](https://ics60.aait.od.ua/zbirnik2024.pdf)

## Supported subjects:
- [Ukrainian language and literature](tests/ukrainian_raw.json) (49 tests with 2746 questions in total)
- [History of Ukraine](tests/history_raw.json) (48 tests with 2640 questions in total)
- [Mathematics](tests/math_raw.json) (9 tests with 230 questions in total)
- [Geography](tests/geography_raw.json) (32 tests with 1788 questions in total)
- English
- German
- French
- Spanish

## TODO:
- [x] Add data files
- [x] Deliver baseline evaluation results (Zero-shot and Chain-of-Thought)

## Evaluation results
- Prompt: TBA
- All evaluations leverage [UA-LLM](https://github.com/NLPForUA/UA-LLM) toolkit

## Learning to "reason" in Ukrainian
#### `google/gemma-2-9b-it` after 4bit PEFT Chain-of-Thought tuning on step-by-step exam solutions ([NLPForUA/gemma-2-it-zno-cot-with-topic](https://huggingface.co/NLPForUA/gemma-2-it-zno-cot-with-topic)): 
[read more](tuning/README.md)
```
<bos><start_of_turn>user
Дайте розгорнуту відповідь на завдання, починаючи з ключового слова "Відповідь:" та використовуючи лише наведені нижче варіанти.

Завдання: З’ясуйте, якими частинами мови є виділені слова в реченні (цифра позначає наступне слово).
Сучасна людина, щоб бути (1)успішною, має вчитися (2)впродовж (3)усього життя, (4)опановуючи нові галузі знань.

Варіанти відповіді:
А – займенник
Б – прикметник
В – форма дієслова (дієприкметник)
Г – форма дієслова (дієприслівник)
Д – прийменник<end_of_turn>
<start_of_turn>model
Відповідь:
ТЕМА: Морфологія. Частини мови
Завдання перевіряє ваше вміння визначати частини мови.
Необхідно бути дуже уважним, тому що частиномовна приналежність конкретного слова часто залежить від контексту. Проаналізуймо кожне із запропонованих слів.
Слово успішною відповідає на питання яка? та вказує на ознаку предмета. Отже, це прикметник.
Слово впродовж не відповідає на питання, а лише поєднує слова в реченні. Отже, це прийменник.
Слово усього відповідає на питання що? і вказує на предмет, але не називає його. Отже, це займенник.
А от слово опановуючи відповідає на питання що роблячи? та вказує на додаткову дію підмета. Отже, це особлива форма дієслова дієприслівник.
Відповідь: 1–Б, 2–Д, 3–А, 4–Г.<end_of_turn>
```

## Raw (parsed) data format:
#### Question with one correct answer:
```json
{
    "task_id": 42,
    "question": "Образ Прометея символізує силу й нескореність народу у творі Тараса Шевченка",
    "answers": [
        {
            "answer": "А",
            "text": "«Сон» («У всякого своя доля...»)"
        },
        {
            "answer": "Б",
            "text": "«Кавказ»"
        },
        {
            "answer": "В",
            "text": "«Гайдамаки»"
        },
        {
            "answer": "Г",
            "text": "«І мертвим, і живим...»"
        },
        {
            "answer": "Д",
            "text": "«Ісаія. Глава 35»"
        }
    ],
    "answer_vheader": [
        "А",
        "Б",
        "В",
        "Г",
        "Д"
    ],
    "answer_hheader": [],
    "correct_answer": [
        "Б"
    ],
    "comment": "",
    "with_photo": false
},
```
#### Question with multiple correct answers (each number from `answer_hheader` corresponds to letter from `answer_vheader`):
```json
{
    "task_id": 48,
    "question": "Укажіть послідовність дій, необхідних для того, щоб зорієнтуватися на місцевості за Полярною зіркою.",
    "answers": [
        {
            "answer": "А",
            "text": "подумки з’єднати прямою лінією дві зірки, що знаходяться на краю «ковша»"
        },
        {
            "answer": "Б",
            "text": "продовжити уявну пряму лінію і відкласти на ній приблизно п’ять відрізків, рівних відстані між двома зірками на краю «ковша»"
        },
        {
            "answer": "В",
            "text": "визначити яскраву зірку в сузір’ї Малої Ведмедиці, яка вказує напрям на північ"
        },
        {
            "answer": "Г",
            "text": "знайти сузір’я Великої Ведмедиці (Великого Возу), яке нагадує ківш"
        }
    ],
    "answer_vheader": [
        "А",
        "Б",
        "В",
        "Г"
    ],
    "answer_hheader": [
        "1",
        "2",
        "3",
        "4"
    ],
    "correct_answer": [
        "Г",
        "А",
        "Б",
        "В"
    ],
    "comment": "",
    "with_photo": false
},
```

### Citation

Please cite the repo if you use the data or code in this repo.

```
@article{zno-eval-2024,
 title={ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian},
 url={https://arxiv.org/abs/2501.06715},
 DOI={10.15276/ict.01.2024.27},
 number={1},
 journal={Informatics. Culture. Technology},
 author={Syromiatnikov, M. V. and Ruvinskaya, V. M. and Troynina, A. S.},
 year={2024},
 month={Sep.}, 
 pages={186-191}
}
```
