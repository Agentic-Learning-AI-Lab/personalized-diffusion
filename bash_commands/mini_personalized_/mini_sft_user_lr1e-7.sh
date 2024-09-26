# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16" train_user.py \
  --train_batch_size=8 \
  --valid_batch_size=8 \
  --max_user_proportion=0.1 \
  --max_gen=500 \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-7 --scale_lr \
  --cache_dir="./pick_a_pic_v2/" \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 32 \
  --tag 0917_mini_sft_user \
  --lora_rank 4 \
  --train_method sft \
  --wandb