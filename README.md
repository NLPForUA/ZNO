# ZNO: Exam questions and answers in Ukrainian

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/NLPForUA/ZNO/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/NLPForUA/ZNO/blob/main/DATA_LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/NLPForUA/ZNO)
![GitHub Repo stars](https://img.shields.io/github/stars/NLPForUA/ZNO?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/NLPForUA/ZNO?style=svg)

This repository contains structured test tasks for multiple subjects from ZNO - Ukrainian External Independent Evaluation (Зовнішнє Незалежне Оцінювання) and NMT - National Multi-Subject Test (Національний Мультипредметний Тест).

Usage and License Notices: The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The code is Apache 2.0 licensed.

## Supported subjects:
- Ukrainian language and literature
- History of Ukraine
- Mathematics
- Geography
- English
- German
- French
- Spanish

## TODO:
- [] Add data files
- [] Deliver baseline evaluation results (Zero-shot and Chain-of-Thought)

## Raw (parsed) data format:
#### Question with one correct answer:
```json
{
    "task_id": 42,
    "question": "Образ Прометея символізує силу й нескореність народу у творі Тараса Шевченка",
    "answers": [
        "А\n«Сон» («У всякого своя доля...»)\nБ\n«Кавказ»\nВ\n«Гайдамаки»\nГ\n«І мертвим, і живим...»\nД\n«Ісаія. Глава 35»"
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
#### Question with multiple correct answers:
```json
{
    "task_id": 48,
    "question": "Укажіть послідовність дій, необхідних для того, щоб зорієнтуватися на місцевості за Полярною зіркою.",
    "answers": [
        "А\nподумки з’єднати прямою лінією дві зірки, що знаходяться на краю «ковша»\nБ\nпродовжити уявну пряму лінію і відкласти на ній приблизно п’ять відрізків, рівних відстані між двома зірками на краю «ковша»\nВ\nвизначити яскраву зірку в сузір’ї Малої Ведмедиці, яка вказує напрям на північ\nГ\nзнайти сузір’я Великої Ведмедиці (Великого Возу), яке нагадує ківш"
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
