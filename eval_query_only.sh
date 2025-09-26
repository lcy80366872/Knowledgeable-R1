#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate noisyrollout

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export GOOGLE_API_KEY="xxx"
#geo3k,hallubench,mathvista,
# Define list of model paths to evaluate
HF_MODEL_PATH="model/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/query_only/"

DATA_DIR="data"

SYSTEM_PROMPT="You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind: 
 - The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
 - Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an exmaple: 
Question: 
what is the capital of China?
<think> reasoning process here </think>
<answer> BeiJing </answer>"

python main_knowledge.py \
  --model $HF_MODEL_PATH \
  --output-dir $RESULTS_DIR \
  --data-path $DATA_DIR \
  --datasets all \
  --tensor-parallel-size 1 \
  --system-prompt="$SYSTEM_PROMPT" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --eval-threads 24 \
  --rag FALSE \
  --device cuda:0 \