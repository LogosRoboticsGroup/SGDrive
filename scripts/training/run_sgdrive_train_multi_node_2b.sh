export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"

TRAIN_TEST_SPLIT=navmini

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
export _CHECK_PEFT=0

export use_world_token="${use_world_token:-True}"
export OCC_TOKEN_NUMBER=${OCC_TOKEN_NUMBER:-625}
export AGENT_TOKEN_NUMBER=${AGENT_TOKEN_NUMBER:-50}
export GP_TOKEN_NUMBER=${GP_TOKEN_NUMBER:-1}
export dream_world="${dream_world:-True}"

export SAVE_MODEL_PATH="/path/to/model_save_dir"
if [ ! -d "$SAVE_MODEL_PATH" ]; then
  mkdir -p "$SAVE_MODEL_PATH"
fi
VLM_PATH='/path/to/internvl_wm_model_save_dir'
CACHE_PATH='/path/to/sgdrive_agent_cache'

torchrun \
    --nnodes ${PET_NNODES} \
    --node_rank ${PET_NODE_RANK} \
    --nproc_per_node ${GPU_NUM_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_sgdrive.py \
    agent=sgdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path=$VLM_PATH \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl_wm" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=1 \
    trainer.params.num_nodes=${PET_NNODES} \
    trainer.params.devices=${GPU_NUM_PER_NODE} \
    experiment_name=training_sgdrive_agent_dit \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    use_cache_without_dataset=True \
    force_cache_computation=False