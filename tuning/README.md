# Empowering Smaller Models: Tuning LLaMA and Gemma with Chain-of-Thought for Ukrainian Exam Tasks

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/NLPForUA/ZNO/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/NLPForUA/ZNO/blob/main/DATA_LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.13988-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2503.13988)

This section contains code and notebooks for Gemma 2 and LLaMA 3.1-3.2 models PEFT tuning on Ukrainian language exam tasks (ZNO and NMT) with and without chain-of-thought output.

## Updates:
- TBA: LLaMA models release on HF
- 2025-03-19: [arXiv preprint](https://arxiv.org/abs/2503.13988)
- 2025-03-18: [Gemma models release on HF](https://huggingface.co/NLPForUA)

## Sample generations of tuned models:

### Sample task:

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
```

### Answer letter only ([NLPForUA/gemma-2-it-zno-al](https://huggingface.co/NLPForUA/gemma-2-it-zno-al)):

```
<start_of_turn>model
Відповідь:
1 - В
2 - Д
3 - А
4 - Г<end_of_turn>
```

### Step-by-step solution ([NLPForUA/gemma-2-it-zno-cot](https://huggingface.co/NLPForUA/gemma-2-it-zno-cot)):

```
<start_of_turn>model
Відповідь:
Завдання перевіряє ваше вміння визначати частини мови.
Необхідно бути дуже уважним, тому що частиномовна приналежність конкретного слова часто залежить від контексту. Проаналізуймо кожне із запропонованих слів.
Слово успішною відповідає на питання якою? і вказує на ознаку, одержану від дії. Отже, це особлива форма дієслова – дієприкметник.
Слово впродовж відповідає на питання де? і вказує на місце дії. Отже, це прийменник.
Слово усього відповідає на питання який? і вказує на ознаку, що не має відношення до предмета. Отже, це займенник.
Слово опановуючи відповідає на питання що роблячи? і вказує на додаткову дію, тому це дієприслівник.
Відповідь: 1–В, 2–Д, 3–А, 4–Г.<end_of_turn>
```

### Step-by-step solution with topic ([NLPForUA/gemma-2-it-zno-cot-with-topic](https://huggingface.co/NLPForUA/gemma-2-it-zno-cot-with-topic)):

```
<start_of_turn>model
Відповідь:
ТЕМА: Морфологія. Частини мови
Завдання перевіряє ваше вміння визначати частини мови.
Необхідно бути дуже уважним, тому що частиномовна приналежність конкретного слова часто залежить від контексту. Проаналізуймо кожне із запропонованих слів.
Слово успішною відповідає на питання яка? та вказує на ознаку предмета. Отже, це прикметник.
Слово впродовж не відповідає на питання, а лише поєднує слова в реченні. Отже, це прийменник.
Слово усього відповідає на питання що? і вказує на предмет. Отже, це займенник.
А от слово опановуючи відповідає на питання що роблячи? та вказує на додаткову дію. Отже, це особлива форма дієслова дієприслівник.
Відповідь: 1 – Б, 2 – Д, 3 – А, 4 – Г.<end_of_turn>
```

### Observation: even though `NLPForUA/gemma-2-it-zno-cot-with-topic` was not trained explicitly on math tasks, it was able to apply the same step-by-step solution paradigm to find the correct answer:

```
<bos><start_of_turn>user
Дайте розгорнуту відповідь на завдання, починаючи з ключового слова "Відповідь:" та використовуючи лише наведені нижче варіанти.

Завдання: Обчисліть площу бічної поверхні правильної трикутної піраміди, сторона основи якої дорівнює 8 см, а апофема на 2 см більша за сторону основи піраміди.

Варіанти відповіді:
А - 72 см^2
Б - 384 см^2
В - 192 см^2
Г - 120 см^2
Д - 240 см^2<end_of_turn>
<start_of_turn>model
Відповідь:
ТЕМА: Форми кривих поверхонь. Стосовний куб. Піраміди.
Завдання перевіряє ваше вміння визначати форму кривих поверхонь, обчислювати площу кривих поверхонь.
Згадаймо, що бічна поверхня піраміди – це сукупність її бічних трикутників.
Площу одного такого трикутника можна обчислити за формулою S = (1/2) * a * l, де a – довжина сторони трикутника, l – довжина його висоти.
У нашому випадку a = 8 см, а l = 8 + 2 = 10 см. Отже, площа одного бічного трикутника дорівнює (1/2) * 8 * 10 = 40 см^2.
Площа бічної поверхні піраміди дорівнює трічі площі одного бічного трикутника, тобто 40 * 3 = 120 см^2.
Відповідь – Г.<end_of_turn>
```

### Citation

Please cite this paper if you use hf models, data or code in this folder.

```
@article{syromiatnikov2025empoweringsmallermodelstuning,
 title={Empowering Smaller Models: Tuning LLaMA and Gemma with Chain-of-Thought for Ukrainian Exam Tasks},
 url={https://arxiv.org/abs/2503.13988},
 DOI={doi.org/10.48550/arXiv.2503.13988},
 number={1},
 journal={},
 author={Mykyta Syromiatnikov and Victoria Ruvinskaya and Nataliia Komleva},
 year={2025},
 month={March}, 
 pages={}
}
```