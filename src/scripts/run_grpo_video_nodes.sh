#!/bin/bash


# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.


cd /mnt/data1/shenghao/M-CoT_3/src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/grpo
mkdir -p ${OUTPUT_DIR_FT}


torchrun --nproc_per_node 8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo.py \
    --output_dir "./log/Qwen2.5-VL-7B-GRPO" \
    --model_name_or_path '/mnt/data1/shenghao/M-CoT_3/src/r1-v/log/Qwen2.5-VL-7B-long-sft38' \
    --dataset_name "/mnt/data1/shenghao/multimodal-cot/Video-R1-data/RL-hard_data17.json" \
    --data_folder "/mnt/data1/shenghao/datasets/" \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 13000 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_iterations 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Video-R1 \
    --report_to none \
    --save_steps 50 \
    --beta 0.0 \
    --max_grad_norm 5 \
    --save_only_model false \
    --freeze_vision_modules true \
    --num_generations 8 \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$RANK.txt && echo "Done."




