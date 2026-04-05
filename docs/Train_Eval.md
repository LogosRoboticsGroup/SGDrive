# SGDrive Training and Evaluation

## Stage 1: Vision-Language Models Driving Pretraining

First, you need to download additional jsonl for QA from HuggingFace:  
(coming soon)

Next, download the **InternVL pretrained weights** from HuggingFace:  
👉 [InternVL3-2B Weights](https://huggingface.co/OpenGVLab/InternVL3-2B)

After downloading, go to `./internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_wm.sh` and configure the training script.  
You can launch the pretraining process with the following commands:

```bash
sh ./internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_wm.sh
```

## Stage 2: Diffusion Planner Imitation Learning

You can download our pretrained **SGDrive VLM** from [SGDrive VLM](coming soon).  

For the diffusion planner training, the first step is to **cache datasets for faster training**.  
Since DiT training converges relatively slowly, training VLM and DiT jointly can be very time-consuming. To accelerate, we cache the hidden states output by the VLM, which enables much faster training.  
> ⚠️ Note: Caching requires approximately **1–2 TB of disk space**. We are also working on faster training methods.  

We also provide the option to skip caching hidden states and directly train VLM + DiT together, though this will be slower. We recommend using SGDrive-2B for training for better efficiency.

### Step 1: Cache hidden states
```bash
# cache dataset (change TRAIN_TEST_SPLIT to navtrain/navtest/navmini for train/test/debug)
sh scripts/cache_dataset/run_caching_sgdrive_hidden_state.sh
```

### Step 2: Configure and run training

Configure the script `scripts/training/run_sgdrive_train_multi_node_2b.sh` and then start training:

```bash
sh scripts/training/run_sgdrive_train_multi_node_2b.sh
```

### Step 3: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.

> ⚠️ **Note:** you **must use NumPy version 1.26.4 or above** to avoid potential errors during metric caching.

```bash
# cache metrics (change TRAIN_TEST_SPLIT to navtrain/navtest/navmini for train/test/debug)
sh scripts/cache_dataset/run_metric_caching.sh
```

### Step 4: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh scripts/evaluation/run_sgdrive_agent_pdm_score_evaluation.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.

## Stage 3: Diffusion Planner Reinforcement Learning Training

In this stage, we perform **reinforcement learning (RL) training** on the Diffusion Planner  to further improve planning performance.

### Step 1: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.

> ⚠️ **Note:** you **must use NumPy version 1.26.4 or above** to avoid potential errors during metric caching.

```bash
# cache metrics (change TRAIN_TEST_SPLIT to navtrain/navtest/navmini for train/test/debug)
sh scripts/cache_dataset/run_metric_caching.sh
```

### Step 2: Configure and Launch RL Training

After caching metrics, configure the RL training script and launch training:

```bash
# Example path to the RL training script
sh scripts/training/run_sgdrive_train_multi_node_2b_rl.sh
```

Before running, modify the script parameters as needed  according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh scripts/evaluation/run_sgdrive_agent_pdm_score_evaluation.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.

