
export RAY_memory_monitor_refresh_ms=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=0
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="xxx"
export SWANLAB_API_KEY="xxx"
MODEL_PATH=model/Qwen2.5-7B-Instruct
EXPERIMENT_NAME=Qwen2.5-7B_ConFiQA_MR_GRPO_w/_RAG
PROJECT_NAME=Knowledgeable-R1
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

SYSTEM_PROMPT="You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind: 
 - The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
 - Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an exmaple: 
Question: 
what is the capital of China?
<think> reasoning process here </think>
<answer> BeiJing </answer>"

python3 -m verl.trainer.main \
    config=training_scripts/config.yaml \
    data.train_files=data/conflict_qa/ConFiQA-MR-train.json \
    data.val_files=data/conflict_qa/ConFiQA-MR-test.json \
    data.rollout_batch_size=512 \
    data.if_augment=false \
    data.max_response_length=2048 \
    data.max_pixels=1000000 \
    data.dataset_name=confiqa \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.use_kl_loss=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.reward.compute_score=math \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.n=32 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.save_freq=9 \

python scripts/model_merger.py --local_dir ${CHECKPOINT_DIR}/global_step_9/actor
mv ${CHECKPOINT_DIR}/global_step_9/actor/huggingface ${CHECKPOINT_DIR}/global_step_9
rm -rf ${CHECKPOINT_DIR}/global_step_9/actor