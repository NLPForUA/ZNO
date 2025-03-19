import os
import random
import unittest
from tools.scoring import compute_all_scores, compute_score, compute_test_scores, compute_val_scores, grade_answer, init_subject_to_grading, restore_answer_codes


class TestGradeAnswer(unittest.TestCase):
    def test_pred_too_long(self):
        """Test that if the predicted answer is longer than 10 characters, score is 0."""
        self.assertEqual(grade_answer("cat", "abcdefghijk", 10), 0)

    def test_single_letter_exact_match(self):
        """For a single-letter correct answer, an exact match should yield max_grade."""
        self.assertEqual(grade_answer("А", "А", 10), 10)

    def test_single_letter_soft_match(self):
        """
        When correct_answer is a single letter and soft=True,
        if the first letter of task_pred matches, return max_grade.
        """
        self.assertEqual(grade_answer("a", "apple", 10, soft=True), 10)

    def test_single_letter_soft_match2(self):
        """
        When correct_answer is a single letter and soft=True,
        if the first letter of task_pred matches, return max_grade.
        """
        self.assertEqual(grade_answer("А", "АВГ", 10, soft=True), 10)

    def test_single_letter_soft_no_match(self):
        """Even with soft=True, if the first letter doesn't match, score should be 0."""
        self.assertEqual(grade_answer("a", "banana", 10, soft=True), 0)

    def test_single_letter_no_match(self):
        """For a single-letter correct answer without soft matching, a non-match returns 0."""
        self.assertEqual(grade_answer("a", "b", 10), 0)

    def test_multi_letter_exact_match(self):
        """For multi-letter answers, an exact match returns max_grade."""
        self.assertEqual(grade_answer("hello", "hello", 10), 10)

    def test_multi_letter_partial_match(self):
        """
        For multi-letter answers, a partial match should return the proportional grade.
        For example, for "cat" vs. "cab", 2 out of 3 letters match.
        """
        expected = 4 * 2 / 3
        result = grade_answer("cat", "cab", 4)
        self.assertAlmostEqual(result, expected)

    def test_multi_letter_partial_match_shorter_pred(self):
        """
        If the predicted answer is shorter than the correct answer,
        only the overlapping portion is considered.
        E.g., "hello" vs. "he" returns 2/5 * max_grade.
        """
        expected = 1 * 2 / 5
        result = grade_answer("АБВГД", "АБ", 1)
        self.assertAlmostEqual(result, expected)

    def test_multi_letter_no_match(self):
        """If none of the letters match in the multi-letter case, the grade should be 0."""
        self.assertEqual(grade_answer("hello", "word", 10), 0)

    def test_both_empty(self):
        """When both the correct answer and task prediction are empty, the function returns 0."""
        self.assertEqual(grade_answer("", "", 10), 0)

    def test_correct_answer_empty(self):
        """An empty correct answer should always return 0, regardless of task_pred."""
        self.assertEqual(grade_answer("", "anything", 10), 0)

    def test_max_task_pred_length_allowed(self):
        """
        The function should allow a task_pred of exactly 10 characters.
        For example, comparing "abcdefghij" with "abcdefgxyz" (7 matching letters)
        should yield a grade of 10 * 7/10.
        """
        expected = 4 * 7 / 10
        result = grade_answer("abcdefghij", "abcdefgxyz", 4)
        self.assertAlmostEqual(result, expected)

    def test_encoding(self):
        """The function should accept Unicode characters in the correct answer and task_pred."""
        self.assertEqual(grade_answer("B", "В", 0), 0)


class TestInitSubjectToGrading(unittest.TestCase):

    def test_single_range(self):
        """
        Test a single subject with a single range.
        For example, "1-3" with a grade of 10 should produce keys 1, 2, and 3 with grade 10.
        """
        input_data = {
            "Math": {
                "1-3": 10
            }
        }
        expected = {
            "Math": {
                1: 10,
                2: 10,
                3: 10
            }
        }
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_multiple_ranges_in_one_subject(self):
        """
        Test a single subject with multiple ranges.
        For example, two ranges "1-2" with grade 5 and "4-5" with grade 8.
        """
        input_data = {
            "Science": {
                "1-2": 5,
                "4-5": 8
            }
        }
        expected = {
            "Science": {
                1: 5,
                2: 5,
                4: 8,
                5: 8
            }
        }
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_multiple_subjects(self):
        """
        Test multiple subjects, each with their own grading ranges.
        """
        input_data = {
            "History": {
                "10-12": 7
            },
            "Geography": {
                "1-1": 9
            }
        }
        expected = {
            "History": {
                10: 7,
                11: 7,
                12: 7
            },
            "Geography": {
                1: 9
            }
        }
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """
        Test that an empty dictionary returns an empty dictionary.
        """
        input_data = {}
        expected = {}
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_empty_inner_dict(self):
        """
        Test that if a subject has an empty grading dictionary,
        the result for that subject is also an empty dictionary.
        """
        input_data = {
            "Literature": {}
        }
        expected = {
            "Literature": {}
        }
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_single_task_range(self):
        """
        Test a range that represents a single task.
        For example, "5-5" should create a single key 5.
        """
        input_data = {
            "Art": {
                "5-5": 12
            }
        }
        expected = {
            "Art": {
                5: 12
            }
        }
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

    def test_invalid_range_format(self):
        """
        Test that an improperly formatted range (i.e. not using '-')
        will raise a ValueError. This test documents the current behavior.
        """
        input_data = {
            "Music": {
                "3_5": 15  # Incorrect delimiter
            }
        }
        with self.assertRaises(ValueError):
            init_subject_to_grading(input_data)

    def test_invalid_range_values(self):
        """
        Test that an improperly formatted range (i.e. not using '-')
        will raise a ValueError. This test documents the current behavior.
        """
        input_data = {
            "Music": {
                "5-4": 15  # Incorrect range values
            }
        }
        expected = {"Music": {}}
        result = init_subject_to_grading(input_data)
        self.assertEqual(result, expected)

class TestRestoreAnswerCodes(unittest.TestCase):
    def test_empty_answers(self):
        """
        If the answers list is empty, the function should return an empty list.
        """
        transformation_mapping = {'A': '1', 'B': '2'}
        answers = []
        result = restore_answer_codes(answers, transformation_mapping)
        self.assertEqual(result, [])

    def test_basic_restoration(self):
        """
        Test that answer codes are correctly restored and the list is sorted
        by the restored code.
        """
        transformation_mapping = {'A': '1', 'B': '2', 'C': '3'}
        answers = [
            {'id': 1, 'answer': '2'},  # should become 'B'
            {'id': 2, 'answer': '3'},  # should become 'C'
            {'id': 3, 'answer': '1'}   # should become 'A'
        ]
        # After restoration: 
        # {'id': 1, 'answer': 'B'}, {'id': 2, 'answer': 'C'}, {'id': 3, 'answer': 'A'}
        # After sorting by 'answer': 'A', 'B', 'C'
        expected = [
            {'id': 3, 'answer': 'A'},
            {'id': 1, 'answer': 'B'},
            {'id': 2, 'answer': 'C'}
        ]
        result = restore_answer_codes(answers, transformation_mapping)
        self.assertEqual(result, expected)

    def test_ordering(self):
        """
        Test that the function returns the answers sorted by the restored 'answer' key.
        """
        transformation_mapping = {'A': 'x', 'B': 'y', 'C': 'z'}
        answers = [
            {'id': 1, 'answer': 'z'},  # becomes 'C'
            {'id': 2, 'answer': 'x'},  # becomes 'A'
            {'id': 3, 'answer': 'y'}   # becomes 'B'
        ]
        expected = [
            {'id': 2, 'answer': 'A'},
            {'id': 3, 'answer': 'B'},
            {'id': 1, 'answer': 'C'}
        ]
        result = restore_answer_codes(answers, transformation_mapping)
        self.assertEqual(result, expected)

    def test_missing_key_in_mapping(self):
        """
        If an answer's code is not found in the reverse mapping, a KeyError should be raised.
        """
        transformation_mapping = {'A': '1', 'B': '2'}
        answers = [
            {'id': 1, 'answer': '3'}  # '3' is not present in the reverse mapping
        ]
        with self.assertRaises(KeyError):
            restore_answer_codes(answers, transformation_mapping)

    def test_additional_fields_untouched(self):
        """
        Ensure that fields other than 'answer' remain unchanged after restoration.
        """
        transformation_mapping = {'Yes': 'Y', 'No': 'N'}
        answers = [
            {'id': 1, 'answer': 'Y', 'comment': 'approved'},
            {'id': 2, 'answer': 'N', 'comment': 'disapproved'}
        ]
        # After restoration: 'Y' -> 'Yes', 'N' -> 'No'
        # Sorted by 'answer': 'No' comes before 'Yes'
        expected = [
            {'id': 2, 'answer': 'No', 'comment': 'disapproved'},
            {'id': 1, 'answer': 'Yes', 'comment': 'approved'}
        ]
        result = restore_answer_codes(answers, transformation_mapping)
        self.assertEqual(result, expected)


class TestComputeScore(unittest.TestCase):

    def test_simple_single_letter_match_no_restore(self):
        """
        Without an answer_restore_map, a simple single-letter prediction should be scored.
        Row prediction contains "Відповідь: B" and correct_answer is ["B"].
        Expect full grade.
        """
        row = {
            "pred": "Some text before Відповідь: B <extra>",
            "input": [{"content": "ignored"}]
        }
        test_task = None  # no restore map
        correct_answer = ["B"]  # a list with a single letter
        task_max_grade = 10
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 10)

    def test_partial_match_multi_letter_no_restore(self):
        """
        Without an answer_restore_map, for multi-letter answers the score should be proportional.
        For correct_answer = ["ABC"] (which becomes "ABC") and prediction "Відповідь: AB",
        expected matching letters = 2 out of 3 so score = 2/3*max_grade.
        """
        row = {
            "pred": "Prefix Відповідь: АБ<ignore>",
            "input": [{"content": "unused"}]
        }
        test_task = None
        correct_answer = ["А", "Б", "В"]
        task_max_grade = 9  # expect 2/3 * 9 = 6
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 6)

    def test_single_letter_with_restore(self):
        """
        With an answer_restore_map and a single-letter correct answer, the map should be applied.
        For instance, if the restore map converts "B" to "C", then the correct answer becomes "C".
        Row prediction is "Відповідь: C", so full grade is awarded.
        """
        row = {
            "pred": "Intro Відповідь: В<other>",
            "input": [{"content": "ignored"}]
        }
        test_task = {"answer_restore_map": {"Б": "В"}, "question": "", "answers": []}
        correct_answer = ["Б"]  # after restore, becomes "C"
        task_max_grade = 5
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 5)

    def test_multi_letter_with_restore_digits(self):
        """
        With an answer_restore_map that contains digit keys and multi-letter correct_answer.
        For example, given:
          test_task["answer_restore_map"] = {"1": "A", "2": "B", "A": "X", "B": "Y"}
        and correct_answer = ["A", "B"] (i.e. "AB"),
        the restoration process creates a mapping:
          For idx 0: map "1" -> value from letter "A" (i.e. "X")
          For idx 1: map "2" -> value from letter "B" (i.e. "Y")
        After sorting by key, correct_answer becomes "XY".
        Prediction should be "Відповідь: XY" so full grade is awarded.
        """
        row = {
            "pred": "Header Відповідь: XY <something>",
            "input": [{"content": "unused"}]
        }
        test_task = {"answer_restore_map": {"1": "A", "2": "B", "A": "X", "B": "Y"}, "question": "", "answers": []}
        correct_answer = ["A", "B"]  # length > 1 triggers digit branch
        task_max_grade = 8
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 8)

    def test_multi_letter_with_restore_digits_exception(self):
        """
        With an answer_restore_map that contains digit keys and multi-letter correct_answer.
        For example, given:
          test_task["answer_restore_map"] = {"1": "A", "2": "B", "A": "X", "B": "Y"}
        and correct_answer = ["A", "B"] (i.e. "AB"),
        the restoration process creates a mapping:
          For idx 0: map "1" -> value from letter "A" (i.e. "X")
          For idx 1: map "2" -> value from letter "B" (i.e. "Y")
        After sorting by key, correct_answer becomes "XY".
        Prediction should be "Відповідь: XY" so full grade is awarded.
        """
        row = {
            "pred": "Header Відповідь: XY <something>",
            "input": [{"content": "Questio"}]
        }
        test_task = {"answer_restore_map": {"1": "A", "2": "B", "A": "X", "B": "Y"}, "question": "Question", "answers": []}
        correct_answer = ["A", "B"]  # length > 1 triggers digit branch
        task_max_grade = 8
        self.assertRaises(AssertionError, compute_score, row, test_task, task_max_grade, correct_answer, debug=False)

    def test_multi_letter_with_restore_digits_exception2(self):
        """
        With an answer_restore_map that contains digit keys and multi-letter correct_answer.
        For example, given:
          test_task["answer_restore_map"] = {"1": "A", "2": "B", "A": "X", "B": "Y"}
        and correct_answer = ["A", "B"] (i.e. "AB"),
        the restoration process creates a mapping:
          For idx 0: map "1" -> value from letter "A" (i.e. "X")
          For idx 1: map "2" -> value from letter "B" (i.e. "Y")
        After sorting by key, correct_answer becomes "XY".
        Prediction should be "Відповідь: XY" so full grade is awarded.
        """
        row = {
            "pred": "Header Відповідь: XY <something>",
            "input": [{"content": "Question: A - text"}]
        }
        test_task = {"answer_restore_map": {"1": "A", "2": "B", "A": "X", "B": "Y"}, "question": "Question", "answers": [{"answer": "A", "text": "text1"}]}
        correct_answer = ["A", "B"]  # length > 1 triggers digit branch
        task_max_grade = 8
        self.assertRaises(AssertionError, compute_score, row, test_task, task_max_grade, correct_answer, debug=False)

        row["input"] = [{"content": "Question: A - text1"}]
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 8)

    def test_prediction_as_list(self):
        """
        When row['pred'] is provided as a list, the function should use the first element.
        """
        row = {
            "pred": ["Prefix Відповідь Б <ignore>"],
            "input": [{"content": "not used"}]
        }
        test_task = None
        correct_answer = ["Б"]
        task_max_grade = 7
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 7)

    def test_prediction_too_long_returns_zero(self):
        """
        When the extracted prediction (after filtering) is longer than 10 characters,
        grade_answer should return 0.
        For example, if prediction is "Відповідь: АБВГД" (i.e. "АБВГД" after filtering),
        score should be 0.
        """
        row = {
            "pred": "Відповідь: А текст текст Відповідь АБВГД<extra>",
            "input": [{"content": "unused"}]
        }
        test_task = None
        # Correct answer is irrelevant since grade_answer returns 0 if task_pred > 10.
        correct_answer = ["А"]
        task_max_grade = 10
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 0)

    def test_wrong_answer_returns_zero(self):
        """
        If the predicted answer (after extraction and filtering) is completely wrong,
        the score should be 0.
        """
        row = {
            "pred": "Prefix Відповідь: Б <ignore>",
            "input": "text text text"
        }
        test_task = None  # no answer_restore_map
        correct_answer = ["А"]
        task_max_grade = 5
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 0)

    def test_no_answer_keyword_returns_zero(self):
        """
        When the keyword "Відповідь" is missing in the prediction,
        extraction fails and the filtered result does not match the correct answer.
        """
        row = {
            "pred": "Completely missing keyword А",
            "input": "text text text"
        }
        test_task = None
        correct_answer = ["A"]
        task_max_grade = 5
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 0)

    # ---- Matching Task Tests ----

    def test_matching_task_full_match_with_numbers(self):
        """
        For matching tasks where the output format is like "1-А 2-Б 3-В 4-Г",
        after filtering, the prediction becomes "АБВГ".
        Given the correct answer as ["А", "Б", "В", "Г"], full grade should be awarded.
        """
        row = {
            "pred": "Intro Відповідь 1-А 2-Б 3-В 4-Г<extra>",
            "input": "text text text"
        }
        test_task = None
        correct_answer = ["А", "Б", "В", "Г"]
        task_max_grade = 4
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 4)

    def test_matching_task_full_match_semicolon(self):
        """
        For matching tasks where the output is provided as "А;Б;В;Г;Д",
        filtering removes semicolons so the prediction becomes "АБВГД".
        With correct answer ["Г", "Б", "А", "В", "Д"] = ["В", "Б", "Д", "Г", "А"],
        and max grade equal to 5,
        2.0 grade should be awarded.
        """
        row = {
            "pred": "Start Відповідь: А;Б;В;Г;Д",
            "input": "text text text",
            "restore_answer_map": {"1": "4", "2": "2", "3": "5", "4": "1", "5": "3"}
        }
        test_task = None
        correct_answer = ["Г", "Б", "А", "В", "Д"]
        task_max_grade = 5
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 2)

    def test_matching_task_partial_match(self):
        """
        For a matching task where the predicted answer is incomplete,
        e.g. correct answer is ["А", "Б", "В", "Г"] but prediction is "А;В;Г" (filtered to "АВГ"),
        only the first letter matches (index 0), so the score should be (1/4)*max_grade.
        """
        row = {
            "pred": "Відповідь Б;Г;Г <irrelevant>",
            "input": "text text text"
        }
        test_task = None
        correct_answer = ["А", "Б", "Г", "Д"]
        task_max_grade = 4
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        # Expected: only "А" matches at index 0, so score = 1/4 * 4 = 1.
        self.assertAlmostEqual(score, 1)

    def test_matching_task_extra_letter_ignored(self):
        """
        If the prediction contains an extra letter, e.g. predicted "АБВГД" while correct answer is ["А", "Б", "В", "Г"],
        the function compares only the first 4 letters.
        In this case, the extra letter is ignored and full score is given.
        """
        row = {
            "pred": "Відповідь: АБВГД <some>",
            "input": "text text text"
        }
        test_task = None
        correct_answer = ["А", "Б", "В", "Г"]
        task_max_grade = 4
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 0)

    def test_matching_task_missing_letter(self):
        """
        If the prediction is missing a letter (e.g. predicted "АБВ" instead of "АБВГ"),
        the matching count will be less, so the score should be proportional.
        For a max grade of 4, if 3 out of 4 letters match, score should be 3.
        """
        row = {
            "pred": "Відповідь: АБВ <end>",
            "input": "text text text"
        }
        test_task = None
        correct_answer = ["А", "Б", "В", "Г"]
        task_max_grade = 4
        score = compute_score(row, test_task, task_max_grade, correct_answer, debug=False)
        self.assertEqual(score, 3)


class TestComputeScoresMethods(unittest.TestCase):

    def setUp(self):
        # For compute_test_scores, we simulate one test file with id "100" and 30 tasks.
        self.test_files_order = ["100"]
        self.num_tasks = 30
        # Create 30 dummy prediction rows.
        self.preds_test = []
        for i in range(self.num_tasks):
            self.preds_test.append({
                "pred": "Prefix Відповідь: A <end>",
                "input": [{"content": "dummy question Answers: А - text1 Б - text2 В - text3 Г - text4 Д - text5"}]
            })
        # Create test_sets: keys "100-0", "100-1", ... "100-29"
        self.test_sets = {}
        for i in range(self.num_tasks):
            self.test_sets[f"100-{i}"] = {
                "correct_answer": ["A"],
                "question": "dummy question",
                "answers": [{"answer": "А", "text": "text1"}, {"answer": "Б", "text": "text2"}, 
                            {"answer": "В", "text": "text3"}, {"answer": "Г", "text": "text4"}, 
                            {"answer": "Д", "text": "text5"}]
            }
        # tasks_grading: assign grade 2 for odd-indexed tasks (1,3,5,...) and 1 for even-indexed tasks.
        self.tasks_grading_test = {i: (2 if i % 2 == 1 else 1) for i in range(1, self.num_tasks+1)}
        
        # For compute_val_scores, create 5 test tasks.
        self.num_val = 5
        self.preds_val = []
        self.test_tasks = []
        for i in range(self.num_val):
            correct_answer = []
            if random.choice([True, False]):
                correct_answer = [random.choice(["А", "Б", "В", "Г"])]
            else:
                correct_answer = random.choices(["А", "Б", "В", "Г", "Д"], k=4)
            # rand letters + "Відповідь: " +
            prefix = "Prefix Відповідь:" + random.choice([" ", ""])
            self.preds_val.append({
                "pred": f"{prefix}{';'.join(correct_answer) if len(correct_answer) > 1 else correct_answer}<|eot|>",
                "input": [{"content": "dummy question Answers: А - text1 Б - text2 В - text3 Г - text4 Д - text5"}]
            })
            self.test_tasks.append({
                "test_id": "val",
                "task_id": i,  # 0-indexed used in compute_val_scores
                "correct_answer": correct_answer,
                "correct_answer_str": ";".join(correct_answer),
                "question": "dummy question",
                "answers": [{"answer": "А", "text": "text1"}, {"answer": "Б", "text": "text2"}, 
                            {"answer": "В", "text": "text3"}, {"answer": "Г", "text": "text4"}, 
                            {"answer": "Д", "text": "text5"}]
            })
        self.tasks_grading_val = {i: (2 if i % 2 == 1 else 1) for i in range(1, self.num_val+1)}
    
    def test_compute_test_scores_no_skip(self):
        # Call compute_test_scores with no skipping.
        scores, _ = compute_test_scores(
            preds=self.preds_test,
            test_sets=self.test_sets,
            tasks_grading=self.tasks_grading_test,
            test_files_order=self.test_files_order,
            skip_grading_for=None,
            debug=False
        )
        # There should be 30 graded tasks for file "100"
        self.assertEqual(len(scores["100"]["all"]), self.num_tasks)
        # Check that tasks with grade >1 (odd-numbered tasks) are in 'matching'
        expected_matching = [self.tasks_grading_test[i+1] for i in range(self.num_tasks) if self.tasks_grading_test[i+1] > 1]
        expected_single = [self.tasks_grading_test[i+1] for i in range(self.num_tasks) if self.tasks_grading_test[i+1] == 1]
        self.assertEqual(len(scores["100"]["matching"]), len(expected_matching))
        self.assertEqual(len(scores["100"]["single"]), len(expected_single))
        # Also, each score should equal the task_max_grade (since the prediction is correct).
        for i, score in enumerate(scores["100"]["all"]):
            expected = self.tasks_grading_test[i+1]
            self.assertEqual(score, expected, f"Task {i+1} expected {expected}, got {score}")

    def test_compute_test_scores_with_skip(self):
        # Skip tasks with indices 5 to 10 (tidx 4 to 9).
        skip = {"100": [[5, 10]]}  # This will skip 6 tasks.
        scores, _ = compute_test_scores(
            preds=self.preds_test,
            test_sets=self.test_sets,
            tasks_grading=self.tasks_grading_test,
            test_files_order=self.test_files_order,
            skip_grading_for=skip,
            debug=False
        )
        # Expect 30 - 6 = 24 graded tasks.
        self.assertEqual(len(scores["100"]["all"]), 24)

    def test_compute_val_scores_no_skip(self):
        # Call compute_val_scores with no skipping.
        scores, _ = compute_val_scores(
            preds=self.preds_val,
            test_tasks=self.test_tasks,
            tasks_grading=self.tasks_grading_val,
            skip_grading_for=None,
            debug=False
        )
        # All 5 tasks should be graded under key "val".
        self.assertEqual(len(scores["val"]["all"]), self.num_val)
        # Check that scores match tasks_grading_val.
        for i, score in enumerate(scores["val"]["all"]):
            expected = self.tasks_grading_val[i+1]
            self.assertEqual(score, expected, f"Test task {i+1} expected {expected}, got {score}")

    def test_compute_val_scores_with_skip(self):
        # Skip tasks in "val" for a range; for example, skip tasks with task_id indices 1 to 3.
        # In compute_val_scores, condition: if tidx >= skip_grading_for[test_idx][0][0]-1 and tidx < skip_grading_for[test_idx][0][1]
        # For test_id "val", if we set skip_grading_for["val"] = [[2,4]], tasks with tidx 1,2,3 will be skipped.
        skip = {"val": [[2, 4]]}
        scores, _ = compute_val_scores(
            preds=self.preds_val,
            test_tasks=self.test_tasks,
            tasks_grading=self.tasks_grading_val,
            skip_grading_for=skip,
            debug=False
        )
        # Out of 5 tasks, skip tasks with indices 1, 2, and 3 => only tasks with tidx 0 and 4 are graded.
        self.assertEqual(len(scores["val"]["all"]), 2)


class TestComputeAllScoresMethod(unittest.TestCase):

    def test_compute_all_scores_cot(self):
        """
        Test the entire scoring process end-to-end.
        """
        # load file from the mock folder
        # compute scores
        # check the results
        
        # get dir of this file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        mock_folder = os.path.join(package_dir, "mock_data")

        result = compute_all_scores(
            folder=os.path.join(mock_folder, "cot-predictions"),
            reference_set=os.path.join(mock_folder, "tasks.json"),
            tasks_grading=None,
            test_files_order=["522", "545"],
            split='test',
            skip_grading_for=None,
            debug=False,
            data_key="test_preds"
        )
        true_scores = {
            'cot-predictions.json': {
                '522': {
                    'all': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2.0, 3.0, 0.0, 0.0, 0.0], 
                    'single': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                    'matching': [2.0, 3.0, 0.0, 0.0, 0.0]
                },
                '545': {
                    'all': [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1.0, 2.0, 1.0, 0.0, 0.0], 
                    'single': [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 
                    'matching': [1.0, 2.0, 1.0, 0.0, 0.0]
                }
            }
        }
        true_average = [{"522": [5, 5.0, 10.0], "545": [8.0, 4.0, 12.0]}, 6.5, 4.5, 11.0]
        scores, average = result
        assert scores == true_scores, f"Expected scores to be {true_scores}, got {scores}"
        assert average == true_average, f"Expected average to be {true_average}, got {average}"

        val_result = compute_all_scores(
            folder=os.path.join(mock_folder, "cot-predictions"),
            reference_set=os.path.join(mock_folder, "val_tasks_522-545.json"),
            tasks_grading=None,
            split='val',
            skip_grading_for=None,
            debug=False,
            data_key="test_preds"
        )

        true_val_average = [{"522": [5, 5.0, 10.0], "545": [8.0, 4.0, 12.0]}, 6.5, 4.5, 11.0, 13.0, 9.0, 22.0]
        val_scores, val_average = val_result
        assert val_scores == true_scores, f"Expected scores to be {true_scores}, got {val_scores}"
        assert val_average == true_val_average, f"Expected average to be {true_average}, got {val_average}"

    def test_compute_all_scores_cot_skip(self):
        """
        Test the entire scoring process end-to-end.
        """
        # load file from the mock folder
        # compute scores
        # check the results
        
        # get dir of this file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        mock_folder = os.path.join(package_dir, "mock_data")

        result = compute_all_scores(
            folder=os.path.join(mock_folder, "cot-predictions"),
            reference_set=os.path.join(mock_folder, "tasks.json"),
            tasks_grading=None,
            test_files_order=["522", "545"],
            split='test',
            skip_grading_for={"522": [[0, 10]], "545": [[5, 10], [25, 27]]},
            debug=False,
            data_key="test_preds"
        )
        true_scores = {
            'cot-predictions.json': {
                '522': {
                    'all': [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2.0, 3.0, 0.0, 0.0, 0.0], 
                    'single': [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                    'matching': [2.0, 3.0, 0.0, 0.0, 0.0]
                },
                '545': {
                    'all': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1.0, 0.0, 0.0], 
                    'single': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                    'matching': [1.0, 0.0, 0.0]
                }
            }
        }
        true_average = [{"522": [4, 5.0, 9.0], "545": [4.0, 1.0, 5.0]}, 4.0, 3.0, 7.0]
        scores, average = result
        assert scores == true_scores, f"Expected scores to be {true_scores}, got {scores}"
        assert average == true_average, f"Expected average to be {true_average}, got {average}"

        val_result = compute_all_scores(
            folder=os.path.join(mock_folder, "cot-predictions"),
            reference_set=os.path.join(mock_folder, "val_tasks_522-545.json"),
            tasks_grading=None,
            split='val',
            skip_grading_for={"522": [[0, 10]], "545": [[5, 10], [25, 27]]},
            debug=False,
            data_key="test_preds"
        )

        true_val_average = [{"522": [4, 5.0, 9.0], "545": [4.0, 1.0, 5.0]}, 4.0, 3.0, 7.0, 8.0, 6.0, 14.0]
        val_scores, val_average = val_result
        assert val_scores == true_scores, f"Expected scores to be {true_scores}, got {val_scores}"
        assert val_average == true_val_average, f"Expected average to be {true_average}, got {val_average}"


    def test_compute_all_scores_no_cot(self):
        """
        Test the entire scoring process end-to-end.
        """
        # load file from the mock folder
        # compute scores
        # check the results
        
        # get dir of this file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        mock_folder = os.path.join(package_dir, "mock_data")

        result = compute_all_scores(
            folder=os.path.join(mock_folder, "no-cot-predictions"),
            reference_set=os.path.join(mock_folder, "tasks.json"),
            tasks_grading=None,
            test_files_order=["545", "568"],
            split='test',
            skip_grading_for=None,
            debug=False,
            data_key="test_preds"
        )
        true_scores = {
            'no-cot-predictions.json': {
                '545': {
                    'all': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 4.0, 0.0, 1.0, 1.0, 1.0], 
                    'single': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], 
                    'matching': [4.0, 0.0, 1.0, 1.0, 1.0]
                },
                '568': {
                    'all': [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 4.0, 1.0, 1.0, 0.0, 1.0], 
                    'single': [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1], 
                    'matching': [4.0, 1.0, 1.0, 0.0, 1.0]
                }
            }
        }
        true_average = [{"545": [7, 7.0, 14.0], "568": [11.0, 7.0, 18.0]}, 9.0, 7.0, 16.0]
        scores, average = result
        assert scores == true_scores, f"Expected scores to be {true_scores}, got {scores}"
        assert average == true_average, f"Expected average to be {true_average}, got {average}"

        val_result = compute_all_scores(
            folder=os.path.join(mock_folder, "no-cot-predictions"),
            reference_set=os.path.join(mock_folder, "val_tasks_545-568.json"),
            tasks_grading=None,
            split='val',
            skip_grading_for=None,
            debug=False,
            data_key="test_preds"
        )

        true_val_average = [{"545": [7, 7.0, 14.0], "568": [11.0, 7.0, 18.0]}, 9.0, 7.0, 16.0, 18.0, 14.0, 32.0]
        val_scores, val_average = val_result
        assert val_scores == true_scores, f"Expected scores to be {true_scores}, got {val_scores}"
        assert val_average == true_val_average, f"Expected average to be {true_average}, got {val_average}"


    def test_compute_all_scores_no_cot_skip(self):
        """
        Test the entire scoring process end-to-end.
        """
        # load file from the mock folder
        # compute scores
        # check the results
        
        # get dir of this file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        mock_folder = os.path.join(package_dir, "mock_data")

        result = compute_all_scores(
            folder=os.path.join(mock_folder, "no-cot-predictions"),
            reference_set=os.path.join(mock_folder, "tasks.json"),
            tasks_grading=None,
            test_files_order=["545", "568"],
            split='test',
            skip_grading_for={"545": [[15, 17]], "568": [[2, 27], [27, 27]]},
            debug=False,
            data_key="test_preds"
        )
        true_scores = {
            'no-cot-predictions.json': {
                '545': {
                    'all': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4.0, 0.0, 1.0, 1.0, 1.0], 
                    'single': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], 
                    'matching': [4.0, 0.0, 1.0, 1.0, 1.0]
                },
                '568': {
                    'all': [1, 1.0, 0.0, 1.0], 
                    'single': [1], 
                    'matching': [1.0, 0.0, 1.0]
                }
            }
        }
        true_average = [{"545": [5, 7.0, 12.0], "568": [1.0, 2.0, 3.0]}, 3.0, 4.5, 7.5]
        scores, average = result
        assert scores == true_scores, f"Expected scores to be {true_scores}, got {scores}"
        assert average == true_average, f"Expected average to be {true_average}, got {average}"

        val_result = compute_all_scores(
            folder=os.path.join(mock_folder, "no-cot-predictions"),
            reference_set=os.path.join(mock_folder, "val_tasks_545-568.json"),
            tasks_grading=None,
            split='val',
            skip_grading_for={"545": [[15, 17]], "568": [[2, 27], [27, 27]]},
            debug=False,
            data_key="test_preds"
        )

        true_val_average = [{"545": [5, 7.0, 12.0], "568": [1.0, 2.0, 3.0]}, 3.0, 4.5, 7.5, 6.0, 9.0, 15.0]
        val_scores, val_average = val_result
        assert val_scores == true_scores, f"Expected scores to be {true_scores}, got {val_scores}"
        assert val_average == true_val_average, f"Expected average to be {true_average}, got {val_average}"

if __name__ == '__main__':
    unittest.main()


