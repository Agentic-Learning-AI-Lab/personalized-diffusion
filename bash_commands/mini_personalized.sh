# python make_sbatch.py --time 48 --bash_files bash_commands/mini_personalized.sh

# pdpo
accelerate launch --mixed_precision="fp16" train_user.py \
  --train_batch_size=8 \
  --valid_batch_size=8 \
  --max_user_proportion=0.1 \
  --max_gen=500 \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-7 --scale_lr \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 32 \
  --tag 0917_mini_pdpo_user \
  --lora_rank 4 \
  --train_method pdpo \
  --wandb

# dpo
accelerate launch --mixed_precision="fp16" train_user.py \
  --train_batch_size=8 \
  --valid_batch_size=8 \
  --max_user_proportion=0.1 \
  --max_gen=500 \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-7 --scale_lr \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 32 \
  --tag 0917_mini_dpo_user \
  --lora_rank 4 \
  --train_method dpo \
  --wandb

# sft
accelerate launch --mixed_precision="fp16" train_user.py \
  --train_batch_size=8 \
  --valid_batch_size=8 \
  --max_user_proportion=0.1 \
  --max_gen=500 \
  --max_train_steps=100000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-7 --scale_lr \
  --allow_tf32 \
  --dataloader_num_worker=16 \
  --gradient_accumulation_steps 32 \
  --tag 0917_mini_sft_user \
  --lora_rank 4 \
  --train_method sft \
  --wandb