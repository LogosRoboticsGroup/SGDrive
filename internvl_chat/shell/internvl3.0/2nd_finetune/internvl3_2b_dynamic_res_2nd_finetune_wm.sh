#!/bin/bash
set -x
set -e

PARTITION=${PARTITION:-"Intern5"}

PET_NNODES=${PET_NNODES:-1}
PET_NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}

GPU_NUM_PER_NODE=${GPU_NUM_PER_NODE:-2}
TOTAL_GPUS=$((PET_NNODES * GPU_NUM_PER_NODE))
BATCH_SIZE=${BATCH_SIZE:-4}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / TOTAL_GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/internvl_chat"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export CUDA_DEVICE_MAX_CONNECTIONS=1

OUTPUT_DIR='exp/internvl_wm_finetune'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun \
  --nnodes ${PET_NNODES} \
  --node_rank ${PET_NODE_RANK} \
  --nproc_per_node ${GPU_NUM_PER_NODE} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  internvl_chat/internvl/train/internvl_chat_finetune_wm.py \
  --model_name_or_path "/path/to/owl10/ReCogDrive-VLM-2B" \
  --conv_style "internvl2_5_world_token" \
  --use_fast_tokenizer False \
  --meta_path "/path/to/sgdrive.json" \
  --force_image_size 448 \
  --max_dynamic_patch 16 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --max_seq_length 12288 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/path/to/zero_stage1_config.json" \
  --report_to "tensorboard" \
  --overwrite_output_dir True \
  --output_dir ${OUTPUT_DIR} \
  --use_world_token True \
  --occ_token_numbers 625 \
  --agent_token_numbers 50 \
  --gp_token_numbers 1 \
  --dream_world True \