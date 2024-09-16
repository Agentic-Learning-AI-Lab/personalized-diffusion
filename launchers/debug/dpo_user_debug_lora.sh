# mainly to debug model saving and loading, just train 2 steps
export MODEL_NAME="pt-sk/stable-diffusion-1.5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16" train_user.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=16 \
  --valid_batch_size=16 \
  --max_user_proportion=0.1 \
  --max_valid_captions=500 \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="./pick_a_pic_v2/" \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 4 \
  --num_inference_steps 1 \
  --tag 0910_dpo_user_debug_lora \
  --checkpointing_steps 2 \
  --lora_rank 4 \
  --wandb
