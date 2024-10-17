# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16"  train.py \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir="train_test_out"
