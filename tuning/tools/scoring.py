from collections import defaultdict
import json
import os
import random
from typing import Dict, List

from tools.prepare_data import hash_dict


SPLITS_PER_SUBJECT = {
    "ukrainian": {
        "test": ["522", "544", "545", "568"],
        "val": ["17", "14", "12", "10", "133", "157", "203", "254", "309", "363", "429", "491", "519"],
        "train": ["15", "16", "25", "31", "32", "13", "30", "11", "19", "6", "4", "130", "132", "172", "175", "178", "139", "143", "168", "189", "206", "240", "281", "299", "335", "347", "389", "401", "436", "452", "471", "515"],
    }
}


NMT_SUBJECT_TO_GRADING = {
    "ukrainian": {
        "1-25": 1,
        "26-30": 4,
    },
    "history": {
        "1-20": 1,
        "21-24": 4,
        "25-27": 4,
        "28-30": 3,
    },
    "geography": {
        "1-36": 1,
        "37-42": 4,
        "43-48": 2,
        "49-54": 3,
    },
}

SKIP_GRADING_FOR = {
    "ukrainian": {
        "17": [[37, 60]],  # [start, end)
        "14": [[37, 60]],  # index starts from 1
        "12": [[37, 60]],
        "10": [[37, 60]],
        "133": [[37, 60]],
        "157": [[34, 57]],
        "203": [[34, 57]],
        "254": [[34, 57]],
        "309": [[34, 57]],
        "363": [[34, 57]],
        "429": [[34, 57]],
        "491": [[1, 24]],
        "519": [],
    }
}

def init_subject_to_grading(subject_to_grading=NMT_SUBJECT_TO_GRADING):
    subject_to_grading_full = {}
    for subject, grades in subject_to_grading.items():
        subject_to_grading_full[subject] = {}
        for tasks, grade in grades.items():
            #print(tasks)
            start, end = map(int, tasks.split("-"))
            for taskid in range(start, end + 1):
                subject_to_grading_full[subject][taskid] = grade
    return subject_to_grading_full


def restore_answer_codes(answers: List[Dict[str, str]], transformation_mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Restores the original answer codes based on a transformation mapping.

    Args:
        answers (List[Dict[str, str]]): List of answers with swapped codes.
        transformation_mapping (Dict[str, str]): Mapping used to transform the codes.

    Returns:
        List[Dict[str, str]]: List of answers with original codes restored.
    """
    reverse_mapping = {v: k for k, v in transformation_mapping.items()}

    # Restore codes
    for a in answers:
        a['answer'] = reverse_mapping[a['answer']]

    return sorted(answers, key=lambda x: x['answer'])


def load_tests(dataset_root_path='D:/research/ZNO/new_tests_v2/ukrainian', subject='ukrainian', splits_per_subject=SPLITS_PER_SUBJECT):
    total_tasks = 0

    loaded_tests = defaultdict(list)

    for file in os.listdir(dataset_root_path):
        if not file.endswith('.json'):
            print("skipped", file)
            continue
        with open(os.path.join(dataset_root_path, file), 'r') as fr:
            test = json.load(fr)
        test_id = file.split('.')[0].split('-')[-1]
        test_tasks = test['tasks']
        test_tasks_clean = []
        for task in test_tasks:
            total_tasks += 1
            task['test_id'] = test_id
            test_tasks_clean.append(task) 
        if test_id in splits_per_subject[subject]['test']:
            loaded_tests[test_id].extend(test_tasks_clean)

    print(len(loaded_tests), total_tasks)
    return load_tests


def grade_answer(correct_answer, task_pred, max_grade, soft=False):
    if len(task_pred) > 10:
        return 0
    if len(correct_answer) == 1:
        if correct_answer == task_pred:
            return max_grade
        elif soft and correct_answer == task_pred[0]:
            return max_grade
        else:
            return 0
    if len(correct_answer) > 1:
        if correct_answer == task_pred:
            return max_grade
        else:
            # find the number of matching letters
            matching_letters = 0
            if not soft and len(task_pred) > len(correct_answer):
                return 0
            for i in range(len(correct_answer)):
                if len(task_pred) <= i:
                    break
                if correct_answer[i] == task_pred[i]:
                    matching_letters += 1
            return matching_letters / len(correct_answer) * max_grade
    return 0


def verify_input_matches_task(row, test_task):
    prediction_input = row['input'][0]['content']
    if isinstance(prediction_input, list):
        prediction_input = prediction_input[0]["text"]
    assert test_task['question'] in prediction_input, f"Model input does not contain the question: {test_task['question']}\n{row['input'][-1]['content']}"
    for i, answer in enumerate(test_task['answers']):
        answer_option = f'{answer["answer"]} - {answer["text"]}'
        assert answer_option in prediction_input, f"Model input does not contain the answer: {answer_option}\n{row['input'][-1]['content']}"
    return True


def compute_score(row, test_task, task_max_grade, correct_answer, soft_grading=False, debug=False):
    if soft_grading:
        print("Warning: soft grading is enabled, the model will receive credit for partially correct answers")
    if isinstance(row['pred'], list):
        row['pred'] = row['pred'][0]
    task_pred = row['pred'].split("Відповідь")[-1].split("<")[0].strip()
    # keep only letters in prediction and answer
    if debug:
        print(test_task)
        print(row['input'][0]['content'])
        print(correct_answer, row['pred'])
    
    if test_task is not None:
        verify_input_matches_task(row, test_task)
    else:
        print("No test task provided for compute_score, make sure it's intended")

    task_pred = ''.join(filter(str.isalpha, task_pred))
    if test_task is not None and "answer_restore_map" in test_task:
        if debug:
            print(correct_answer, test_task["answer_restore_map"])
        if len(correct_answer) > 1 and any(k.isdigit() for k in test_task["answer_restore_map"]):
            correct_answer_map = {}
            for idx, letter in enumerate(correct_answer):
                correct_answer_map[test_task["answer_restore_map"][str(idx+1)]] = test_task["answer_restore_map"][letter]
            if debug:
                print(correct_answer_map)
            correct_answer = sorted(correct_answer_map.items(), key=lambda x: x[0])
            if debug:
                print(correct_answer)
            correct_answer = ''.join([c[1] for c in correct_answer])
        else:
            correct_answer = ''.join([test_task["answer_restore_map"][c] for c in correct_answer])
    else:
        correct_answer = ''.join(filter(str.isalpha, correct_answer))

    curr_score = grade_answer(correct_answer, task_pred, task_max_grade, soft=soft_grading)
    if debug:
        print(correct_answer, task_pred, curr_score, '\n\n')
    return curr_score


def compute_test_scores(
    preds,
    test_sets,
    tasks_grading,
    test_files_order=["522", "568", "545", "544"],
    skip_grading_for=None,
    debug=False,
):
    scores = defaultdict(list)
    skipped_tasks = []

    for i, test_idx in enumerate(test_files_order):
        scores[test_idx] = {"all": [], "single": [], "matching": []}
        for tidx, row in enumerate(preds[i*30:(i+1)*30]):
            skip_task = False
            if skip_grading_for and skip_grading_for[test_idx]:
                for skip_span in skip_grading_for[test_idx]:
                    if isinstance(skip_span, dict):
                        if tidx == skip_span["task_id"] and test_idx == skip_span["test_id"]:
                            verify_input_matches_task(row, skip_span)
                            if debug:
                                print(f"\n\nSkipping task {skip_span['task_id']} in test {skip_span['test_id']}\nwith question: {skip_span['question'][:50]}\n\n")
                            skip_task = True
                            break
                    else:
                        assert len(skip_span) == 2
                        if tidx < (skip_span[0] - 1) or tidx >= skip_span[1]:
                            continue
                        if tidx >= (skip_span[0] - 1) and tidx < skip_span[1]:
                            skip_task = True
                            break

            test_task = None
            if test_sets is not None:
                test_task = test_sets[f"{test_idx}-{tidx}"]

            if skip_task:
                if test_task is not None:
                    skipped_tasks.append(test_task)
                else:
                    skipped_tasks.append(row)
                continue

            if tasks_grading is not None:
                task_max_grade = tasks_grading[tidx + 1]
            else:
                task_max_grade = len(test_task['correct_answer'])

            assert isinstance(test_task['correct_answer'], list) == True

            correct_answer = test_task['correct_answer']
            curr_score = compute_score(row, test_task, task_max_grade, correct_answer, debug=debug)
            
            scores[test_idx]['all'].append(curr_score)
            if task_max_grade > 1:
                scores[test_idx]['matching'].append(curr_score)
            else:
                scores[test_idx]['single'].append(curr_score)

    return scores, skipped_tasks


def compute_val_scores(
    preds,
    test_tasks,
    tasks_grading=None,
    skip_grading_for=None,
    debug=False
):
    scores = defaultdict(lambda: {"all": [], "single": [], "matching": []})

    skipped_tasks = []

    for i, test_task in enumerate(test_tasks):
        row = preds[i]
        test_idx = test_task['test_id']
        tidx = test_task['task_id']
        
        skip_task = False
        if skip_grading_for and skip_grading_for[test_idx]:
            for skip_span in skip_grading_for[test_idx]:
                if isinstance(skip_span, dict):
                    if tidx == skip_span["task_id"] and test_idx == skip_span["test_id"]:
                        verify_input_matches_task(row, skip_span)
                        if debug:
                            print(f"\n\nSkipping task {skip_span['task_id']} in test {skip_span['test_id']}\nwith question: {skip_span['question'][:50]}\n\n")
                        skip_task = True
                        break
                else:
                    if tidx < (skip_span[0] - 1) or tidx >= skip_span[1]:
                        continue
                    if tidx >= (skip_span[0] - 1) and tidx < skip_span[1]:
                        skip_task = True
                        break

        if skip_task:
            skipped_tasks.append(test_task)
            continue

        if tasks_grading is not None:
            task_max_grade = tasks_grading[tidx + 1]
        else:
            task_max_grade = len(test_task['correct_answer'])

        assert isinstance(test_task['correct_answer'], list) == True

        correct_answer = test_task['correct_answer']
        curr_score = compute_score(row, test_task, task_max_grade, correct_answer, debug=debug)
            
        scores[test_idx]['all'].append(curr_score)
        if task_max_grade > 1:
            scores[test_idx]['matching'].append(curr_score)
        else:
            scores[test_idx]['single'].append(curr_score)

    return scores, skipped_tasks


def compute_all_scores(
    folder,
    reference_set,
    tasks_grading,
    test_files_order=["522", "568", "545", "544"],
    split='test',
    skip_grading_for=None,
    debug=False,
    data_key=None,
    return_skipped=False,
    strict_validation=True,
    output_sum=False,
):
    scores = {}
    skipped_tasks = {}
    test_tasks = None
    task_order = []
    test_ids = set()
    skip_grading_for_hash = None
    if reference_set is not None:
        test_tasks = {} if split == "test" else []
        with open(reference_set, 'r', encoding="utf-8") as fr:
            reference_set = json.load(fr)
            for row in reference_set[split]:
                test_ids.add(row['test_id'])
                if row['test_id'] not in task_order:
                    task_order.append(row['test_id'])
                if split == "test":
                    test_tasks[f"{row['test_id']}-{row['task_id']}"] = row
                else:
                    test_tasks.append(row)
    
    if isinstance(skip_grading_for, str):
        with open(skip_grading_for, 'r', encoding="utf-8") as fr:
            skip_grading_for_data = json.load(fr)
        # keep only the tests that are in the reference set
        skip_grading_for_subset = []
        skip_grading_for = defaultdict(list)
        for skip_task in skip_grading_for_data:
            if skip_task['test_id'] in test_ids:
                skip_grading_for_subset.append(skip_task)
                skip_grading_for[skip_task['test_id']].append(skip_task)
        print(f"Keeping only {len(skip_grading_for_subset)} out of {len(skip_grading_for_data)} tasks for grading skipping")
        # sort and get hash
        skip_grading_for_sorted = sorted(skip_grading_for_subset, key=lambda x: f'{x["test_id"]}_{x["task_id"]}: {x["question"]}')
        skip_grading_for_hash = hash_dict(skip_grading_for_sorted)

    
    for file in os.listdir(folder):
        if not file.endswith('.json'):
            continue
        experiment_link = None
        with open(os.path.join(folder, file), 'r', encoding="utf-8") as fr:
            preds = json.load(fr)
            if "configs" in preds:
                experiment_link = preds['configs'].get('experiment_link', None)
            if data_key is not None and data_key in preds:
                preds = preds[data_key]
        if split == "test":
            compute_result = compute_test_scores(preds, test_tasks, tasks_grading, test_files_order, skip_grading_for, debug=debug)
        else:
            compute_result = compute_val_scores(preds, test_tasks, tasks_grading, skip_grading_for, debug=debug)
        scores[file], skipped_tasks[file] = compute_result[0], compute_result[1]

        skipped_tasks_sorted = sorted(skipped_tasks[file], key=lambda x: f'{x["test_id"]}_{x["task_id"]}: {x["question"]}')
        skipped_tasks_hash = hash_dict(skipped_tasks_sorted)
        if strict_validation:
            if skip_grading_for is not None and skip_grading_for_hash is not None:
                assert skip_grading_for_hash == skipped_tasks_hash, f"Actual skipped tasks do not match the provided skip_grading_for file for {file}"

        
        output = [
            {test_idx: [sum(scores[file][test_idx]['single']), sum(scores[file][test_idx]['matching']), sum(scores[file][test_idx]['all'])] for test_idx in scores[file]},
            sum(sum(v['single']) for v in scores[file].values())/len(scores[file]),
            sum(sum(v['matching']) for v in scores[file].values())/len(scores[file]),
            sum(sum(v['all']) for v in scores[file].values())/len(scores[file])
        ]
        num_skipped = len(skipped_tasks[file])
        if split == "val" or output_sum:
            output += [
                sum(sum(v['single']) for v in scores[file].values()),
                sum(sum(v['matching']) for v in scores[file].values()),
                sum(sum(v['all']) for v in scores[file].values())
            ]
        if experiment_link is not None:
            print(f"\n{experiment_link}\n{file}\nskipped: {num_skipped}\n{output}")
        else:
            print(file, output)
        
        print(f"Skip_grading_for_hash: {skip_grading_for_hash}, skipped_tasks_hash: {skipped_tasks_hash}\n")
    
    if not return_skipped:
        return scores, output
    return scores, output, skipped_tasks


def compute_random_guess_scores(
    loaded_tests: Dict[str, List[Dict]],
    subject_to_grading_full: Dict[str, Dict],
    subject: str = "ukrainian"
) -> Dict[str, Dict[str, List[float]]]:
    """
    `loaded_tests` is a dictionary with test indices as keys and lists of tasks as values.
    `subject_to_grading_full` is a dictionary with subject names as keys and dictionaries as values,
    where the inner dictionaries have task indices as keys and grades as values.
    """
    scores = {}

    for i, test_idx in enumerate(["522", "568", "545", "544"]):
        scores[test_idx] = {"all": [], "single": [], "matching": []}
        for tidx, task in enumerate(loaded_tests[test_idx]):
            #print(loaded_tests[test_idx][tidx]['correct_answer'], '\n\n', row['pred'].split("Відповідь")[-1], '\n\n\n')
            correct_answer = loaded_tests[test_idx][tidx]['correct_answer']
            if not task['answer_hheader']:
                task_pred = random.choice(task['answer_vheader'])
            else:
                # randomly place len(task['vheader']) letters in the answer
                task_pred = list(task['answer_vheader'])
                random.shuffle(task_pred)
                task_pred = ''.join(task_pred[:len(task['answer_hheader'])])

            # keep only letters in prediction and answer
            task_pred = ''.join(filter(str.isalpha, task_pred))
            correct_answer = ''.join(filter(str.isalpha, correct_answer))
            #print(correct_answer, task_pred)
            task_max_grade = subject_to_grading_full[subject][tidx + 1]
            curr_score = grade_answer(correct_answer, task_pred, task_max_grade)
            if task_max_grade > 1:
                scores[test_idx]["matching"].append(curr_score)
            else:
                scores[test_idx]["single"].append(curr_score)
            scores[test_idx]["all"].append(curr_score)
    return scores


def calculate_random_stats(loaded_tests: Dict[str, List[Dict]], subject_to_grading_full: Dict[str, Dict], subject: str = "ukrainian", n_iter: int = 50000):
    rand_scores_per_test = defaultdict(list)
    for i in range(n_iter):
        random_scores = compute_random_guess_scores(loaded_tests, subject_to_grading_full, subject)
        for test_idx in random_scores:
            if not test_idx in rand_scores_per_test:
                rand_scores_per_test[test_idx] = {"all": [], "single": [], "matching": []}
            for t in random_scores[test_idx]:
                rand_scores_per_test[test_idx][t].append(sum(random_scores[test_idx][t]))
    all_res = {test_idx: sum(rand_scores_per_test[test_idx]['all'])/len(rand_scores_per_test[test_idx]['all']) for test_idx in rand_scores_per_test}
    single_res = {test_idx: sum(rand_scores_per_test[test_idx]['single'])/len(rand_scores_per_test[test_idx]['single']) for test_idx in rand_scores_per_test}
    matching_res = {test_idx: sum(rand_scores_per_test[test_idx]['matching'])/len(rand_scores_per_test[test_idx]['matching']) for test_idx in rand_scores_per_test}
    print(all_res, sum(v for v in all_res.values())/len(all_res), single_res, sum(v for v in single_res.values())/len(single_res), matching_res, sum(v for v in matching_res.values())/len(matching_res))