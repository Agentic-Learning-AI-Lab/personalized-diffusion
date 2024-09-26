export MODEL_NAME="pt-sk/stable-diffusion-1.5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16" train_general.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --valid_batch_size=8 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="./pick_a_pic_v2/" \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 128 \
  --max_train_samples 100000 \
  --tag 0914_dpo_2gpu_lora \
  --lora_rank 4 \
  --wandb

# 80hrs on 2 A100