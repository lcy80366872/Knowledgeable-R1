import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
from verl.utils.reward_score.qa_em import em_check,subem_check,normalize_answer
from mathruler.grader import extract_boxed_content, grade_answer

def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        
      
        # Create prompt
        if args.rag=="TRUE":
            query =item["passage"]+item["problem"]
        else:
            query = item["problem"]
        prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                
        
        prompts.append({
            "prompt": prompt_text,
        })
        
        metadata.append({
            "dataset": dataset_name,
            "question": item["problem"],
            "answer": item["answer"],
            "prompt": prompt_text,
            
        })
        if "is_cf" in item:
            metadata[-1]["is_cf"] = item["is_cf"]
    
    return prompts, metadata
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s
def parse_answer(text: str) -> str:
    
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None

def exact_match_score(prediction, ground_truth):
    # 统一处理ground_truth为列表形式
    truths = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    normalized_pred = normalize_answer(bool_mapping(prediction))
    
    for truth in truths:
        if normalized_pred == normalize_answer(bool_mapping(truth)):
            return True
    return False
def evaluate_prediction(prediction: str, answer: str, dataset: str, question: str = "") -> float:
    """Evaluate a prediction against the ground truth"""
    extracted_answer = parse_answer(prediction)
    if extracted_answer is None:
        return 0.0
    return 1.0 if exact_match_score(extracted_answer, answer) else 0.0
    

def process_outputs(outputs, metadata, max_workers: int) -> Dict[str, List[Dict]]:
    """Process model outputs and calculate metrics"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            meta = metadata[i]
            dataset = meta["dataset"]
            if "question_for_eval" in meta:
                question = meta["question_for_eval"]
            else:
                question = meta["question"]
            
            future = executor.submit(
                evaluate_prediction, 
                prediction, 
                meta["answer"], 
                dataset,
                question
            )
            futures.append((future, i, prediction, meta))
        
        for future, i, prediction, meta in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
                
                result = {
                    
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "correct": accuracy > 0,
                    **{k: v for k, v in meta.items() if k not in ["dataset","question", "answer"]}
                }
                
                results.append(result)
            except Exception as e:
                print(f"Error evaluating prediction {i}: {str(e)}")
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    if not results:
        return {"accuracy": 0.0}
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    metrics = {"accuracy": accuracy}
    
    # Calculate task-specific accuracies if available
    if any("task" in r for r in results):
        task_results = {}
        for r in results:
            if "task" in r:
                task = r["task"]
                if task not in task_results:
                    task_results[task] = []
                task_results[task].append(r["correct"])
        
        task_accuracies = {task: sum(results) / len(results) for task, results in task_results.items()}
        metrics["sub_accuracies"] = task_accuracies
    
    # Calculate problem version accuracies if available
    if any("problem_version" in r for r in results):
        version_results = {}
        for r in results:
            if "problem_version" in r:
                version = r["problem_version"]
                if version not in version_results:
                    version_results[version] = []
                version_results[version].append(r["correct"])
        
        version_accuracies = {version: sum(results) / len(results) for version, results in version_results.items()}
        metrics["sub_accuracies"] = version_accuracies
    
    # Calculate subject accuracies if available
    if any("subject" in r for r in results):
        subject_results = {}
        for r in results:
            if "subject" in r:
                subject = r["subject"]
                if subject not in subject_results:
                    subject_results[subject] = []
                subject_results[subject].append(r["correct"])
        
        subject_accuracies = {subject: sum(results) / len(results) for subject, results in subject_results.items()}
        metrics["sub_accuracies"] = subject_accuracies
    
    return metrics