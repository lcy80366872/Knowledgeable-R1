# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")
    def remove_commas_from_numbers(text):
        # Use regex to find numbers with commas and remove the commas
        return re.sub(r'(?<=\d),(?=\d)', '', text)
    return white_space_fix(remove_articles(remove_punc(lower(remove_commas_from_numbers(replace_underscore(s))))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    return all(ground in pre_list for ground in ground_list)


def exact_match_score(prediction, ground_truth):
    truths = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    normalized_pred = normalize_answer(bool_mapping(prediction))
    
    for truth in truths:
        if normalized_pred == normalize_answer(bool_mapping(truth)):
            return 1
    return 0

    

def cau_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = 0

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em_check(golden_answer,prediction):
    normalized_prediction = normalize_answer(prediction)
    score = -1
    golden_answer = normalize_answer(golden_answer)
    if golden_answer == normalized_prediction:
        score = 1
    return score

def subem_check(golden_answer,prediction):
    normalized_prediction = normalize_answer(prediction)
    score = -1
    golden_answer = normalize_answer(golden_answer)
    if golden_answer in normalized_prediction:
        score = 1
    return score

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
 

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
def pure_acc_compute_score(predict_str: str, ground_truth: str) -> float:
    return em_reward(predict_str, ground_truth)
def qa_acc_compute_score(predict_str: str, ground_truth: str) -> float:
    # print(f"--------------------------------")
    # print(0.45*em_reward(predict_str, ground_truth))
    # print(0.45*f1_reward(predict_str, ground_truth))
    # print(0.1*qa_format_reward(predict_str))
    return 0.45*em_reward(predict_str, ground_truth)+0.45*f1_reward(predict_str, ground_truth)+0.1*qa_format_reward(predict_str)
def qa_format_reward(answer)-> float:
    pattern = r"^<think>.*?</think>[\n ]<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    reward = 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else 0
    return reward
def parse_answer(text: str) -> str:
    match = re.search(r'.*<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ''
def em_reward(predict_str: str, ground_truth: str) -> float:
    answer = parse_answer(predict_str)
    score =exact_match_score( answer,ground_truth)
    return score
def f1_reward(predict_str: str, ground_truth: str) -> float:
    answer = parse_answer(predict_str)
    return  cau_f1_score( answer,ground_truth)