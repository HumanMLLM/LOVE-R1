#!/bin/bash


cd /mnt/data1/shenghao/M-CoT/src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/sft_stage2
mkdir -p ${OUTPUT_DIR_FT}


torchrun --nproc_per_node 8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/sft_video.py \
    --output_dir "./log/Qwen2.5-VL-7B-sft-stage2" \
    --model_name_or_path "/mnt/data/shenghao/LongVideoR1/M-CoT_2/src/r1-v/log/Qwen2.5-VL-7B-sft36" \
    --dataset_name "/mnt/data/shenghao/LongVideoR1/zoomin_cot_data_0904_38k_filtered.json" \
    --data_folder "/mnt/data/shenghao/datasets/" \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name Qwen2.5-VL-7B-sft-stage2 \
    --save_steps 50 \
    --max_grad_norm 5 \
    --dataloader_prefetch_factor 2 \
    --dataloader_num_workers 2 \
    --freeze_vision_modules true \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$RANK.txt && echo "Done."

