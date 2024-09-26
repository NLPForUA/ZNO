# ZNO-Eval: Exam questions and answers in Ukrainian

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/NLPForUA/ZNO/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/NLPForUA/ZNO/blob/main/DATA_LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/NLPForUA/ZNO)
![GitHub Repo stars](https://img.shields.io/github/stars/NLPForUA/ZNO?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/NLPForUA/ZNO?style=svg)

This repository contains structured test tasks for multiple subjects from ZNO - Ukrainian External Independent Evaluation (Зовнішнє Незалежне Оцінювання) and NMT - National Multi-Subject Test (Національний Мультипредметний Тест).

Usage and License Notices: The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The code is Apache 2.0 licensed.

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
- [] Add data files
- [] Deliver baseline evaluation results (Zero-shot and Chain-of-Thought)

## Evaluation results
- Prompt: TBA
- All evaluations leverage [UA-LLM](https://github.com/NLPForUA/UA-LLM) toolkit

### Math (NMT - National multi-subject test)
#### Zero-shot evaluation
| Model | Score[^1] | ZNO Score (100-200)[^1] |
| --- | :---: | :---: |
| `gpt-3.5-turbo-0125` | 7.5 | 96.5 (failed) |
| `mistral-large-latest` | 11.75 | 139.0 |
| `claude-3-opus-20240229` | 17.25 | 149.25 |
| `gpt-4o-2024-05-13` | 17.25 | 149.25 |
| `gpt-4-turbo-2024-04-09` | 20.0 | 152.75 |
| `gemini-1.5-pro-preview-0514` | **20.5** | **155.75** |

[^1]: average of 4 tests

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
@misc{zno-eval-2024,
  author = {Mykyta Syromiatnikov},
  title = {ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NLPForUA/ZNO}},
}
```
