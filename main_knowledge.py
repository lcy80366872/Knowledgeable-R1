import argparse
import json
import os
import torch
from vllm import LLM, SamplingParams
from data_process.musique import musique
from data_process.hotpot import hotpotqa
from data_process.choice import choice
from data_process.conflict_qa import conflict,conflict_mix
from data_process.processing_knowledge import (
    prepare_prompts,
    process_outputs,
    calculate_metrics
)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    
    # Model and runtime parameters
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--eval-threads", type=int, default=32, help="Number of threads for evaluation")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    parser.add_argument("--version", type=str, default="7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rag", type=str, default="TRUE")
    # Dataset selection
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'")
    
    # Dataset-specific paths
    parser.add_argument("--data-path", type=str, default="data", help="")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    device =args.device
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    max_results =5   #retrieve top-k contexts
    # Determine which datasets to evaluate
    if args.datasets == "qa":
        datasets_to_eval = ['hotpotqa', 'musique', '2wiki']
    elif args.datasets == "confiqa":
        datasets_to_eval = ['confiqa_qa', 'confiqa_mr', 'confiqa_mc','confiqa_sc']
    elif args.datasets == "all":
        datasets_to_eval = [
            'hotpotqa', 'musique','2wiki','explainpe', 'confiqa_qa', 'confiqa_mr', 'confiqa_mc','confiqa_sc'
        ]
    else:
        datasets_to_eval = args.datasets.split(",")
    
    
    # Dictionary to store all samples
    all_samples = {}
    
    # Load datasets based on selection
    for dataset_name in datasets_to_eval:
        if dataset_name == 'hotpotqa':
            DATASET_PATH = args.data_path+"/hotpot/hotpot_dev_fullwiki_v1.json"
            dataset_process= hotpotqa
            all_samples['hotpotqa'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['hotpotqa'])} samples from HotpotQA")
        elif dataset_name == '2wiki':
            DATASET_PATH = args.data_path+"/2wiki/dev.json"
            dataset_process= hotpotqa
            all_samples['2wiki'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['2wiki'])} samples from 2WikiMultiHopQA")
        elif dataset_name == 'musique':
            DATASET_PATH = args.data_path+"/MuSiQue/musique_ans_v1.0_dev.jsonl"
            dataset_process= musique
            all_samples['musique'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['musique'])} samples from Musique")
        elif dataset_name == 'explainpe':
            DATASET_PATH = args.data_path+"/explainpe/explainpe_test.xlsx"
            dataset_process= choice
            all_samples['explainpe'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['explainpe'] )} samples from explainpe")
        elif dataset_name == 'confiqa_qa':
            DATASET_PATH = args.data_path+"/conflict_qa/ConFiQA-QA-test.json"
            dataset_process= conflict
            all_samples['confiqa_qa'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['confiqa_qa'])} samples from ConFiQA-QA")
        elif dataset_name == 'confiqa_mr':
            DATASET_PATH = args.data_path+"/conflict_qa/ConFiQA-MR-test.json"
            dataset_process= conflict
            all_samples['confiqa_mr'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['confiqa_mr'])} samples from ConFiQA-MR")
        elif dataset_name == 'confiqa_mc':
            DATASET_PATH = args.data_path+"/conflict_qa/ConFiQA-MC-test.json"
            dataset_process= conflict
            all_samples['confiqa_mc'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['confiqa_mc'])} samples from ConFiQA-MC")
        elif dataset_name == 'confiqa_sc':
            DATASET_PATH = args.data_path+"/conflict_qa/ConFiQA-MIX-test.json"
            dataset_process= conflict_mix
            all_samples['confiqa_sc'] = dataset_process(DATASET_PATH,max_results)
            print(f"Loaded {len(all_samples['confiqa_sc'] )} samples from ConFiQA-SC")
        
    
    if not all_samples:
        print("No datasets loaded. Please check the paths and dataset names.")
        return
    
    # Initialize model
    print(f"Initializing model from {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.7,
        device=device, 
        max_model_len=args.max_model_len
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # Process in batches
    all_results = {}
    for dataset_name in all_samples.keys():
        all_results[dataset_name] = []
    
    for dataset_name, samples in all_samples.items():
        prompts, metadata = prepare_prompts(dataset_name, samples, args)
        
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs
        
        results = process_outputs(outputs, metadata, args.eval_threads)
        if "document" in results:
            results.pop("document")
        all_results[dataset_name] = results
        
        metrics = calculate_metrics(results)
        
        output_dict = {
            "results": results,
            "metrics": metrics,
            "config": vars(args)
        }
        
        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
        print(f"{dataset_name.upper()} Results:")
        print(f"  Total samples: {len(results)}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'sub_accuracies' in metrics:
            print("  Task/Category Accuracies:")
            for task, acc in metrics['sub_accuracies'].items():
                print(f"    {task}: {acc:.4f}")
        print()
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()