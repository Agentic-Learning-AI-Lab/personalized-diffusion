# TODO: SDXL
# TODO: multi-GPU evaluation

#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import math
from collections import defaultdict, Counter

import datasets
import numpy as np
from PIL import Image
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers

from utils.misc import cast_training_params, chunks, compute_loss_sft, compute_loss_dpo, count_param_in_model
from utils.pickscore_utils import Selector as PickScoreSelector

import wandb
wandb.login(key='faf21d9ff65ee150697c7e96f070616f6b662134', relogin=True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--lora_rank", type=int, default=-1, help="The dimension of the LoRA update matrices.")
    # data
    parser.add_argument("--dataset_name", type=str, default='yuvalkirstain/pickapic_v2')
    parser.add_argument("--cache_dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--max_valid_captions", type=int, default=None, help="For debugging purposes or quicker evaluation, truncate the number of evaluation examples.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images")
    parser.add_argument("--random_crop", default=False, action="store_true", help="If set the images will be randomly cropped (instead of center), images resized before cropping")
    parser.add_argument("--no_hflip", action="store_true", help="whether to supress horizontal flipping")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    # training
    parser.add_argument("--max_train_steps", type=int, default=2000, help="Total number of training steps to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming")
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-8, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", help='The scheduler type from ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # logging
    parser.add_argument("--tag", type=str, default=None, required=True, help='tag for output dir and wandb run')
    parser.add_argument("--output_dir", type=str, default="output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--wandb", action='store_true', help="whether to log wandb")
    parser.add_argument("--tracker_project_name", type=str, default="personalized_diffusion", help="The `project_name` argument passed to tor.init_trackers")
    # DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0, help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    # misc
    parser.add_argument("--seed", type=int, default=831, help="A seed for reproducible training.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help='whether to use mixed precision training')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # eval
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--n_log_valid_imgs", type=int, default=10, help="Number generated images to log in evaluation")
    # user preference training
    parser.add_argument("--max_user_proportion", type=float, default=1.0, help='proportion of user to keep, used to subset our dataset')
    parser.add_argument("--per_user_contribution_thr", type=int, default=200, help='users with less contribution than this threshold are put into validation')
    parser.add_argument("--per_user_n_eval", type=int, default=100, help='number of evaluation samples per user')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.train_method = 'sft' if args.sft else 'dpo'
    args.output_dir = os.path.join(args.output_dir, args.tag)

    assert args.per_user_n_eval < args.per_user_contribution_thr
    assert 0.0 <= args.max_user_proportion <= 1.0
    return args


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    args = parse_args()
    assert 'pickapic' in args.dataset_name # just for now

    #### START ACCELERATOR BOILERPLATE ###
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb' if args.wandb else None,
        project_config=accelerator_project_config,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    ### END ACCELERATOR BOILERPLATE

    def save_model(unet, save_path):
        if args.lora_rank > 0:
            logger.info('saving lora model')
            unwrapped_unet = accelerator.unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_path,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
        else:
            logger.info('saving unet model (no lora)')
            accelerator.save_state(save_path)

    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weight_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae', torch_dtype=weight_dtype)
    # clone of model
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype)

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)

    # unet & optionally LORA
    unet_dtype = weight_dtype if args.lora_rank > 0 else torch.float32
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=unet_dtype)
    if args.lora_rank > 0:
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        if args.mixed_precision == 'fp16':
            cast_training_params(unet, dtype=torch.float32)
    unet.enable_xformers_memory_efficient_attention()
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters()) if args.lora_rank > 0 else None

    # load PickScore as the selector model (for evaluation only for now)
    pickscore_model = PickScoreSelector(accelerator.device)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.AdamW(
        lora_layers if args.lora_rank > 0 else unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples['caption']:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `caption` should contain either strings or lists of strings.")
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    valid_transforms = transforms.Compose( # always center crop, no flip
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ##### START BIG OLD DATASET BLOCK #####
    #### START PREPROCESSING/COLLATION ####

    def preprocess_data_dpo(examples, transforms):
        all_pixel_values = []
        for col_name in ['jpg_0', 'jpg_1']:
            images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            pixel_values = [transforms(image) for image in images]
            all_pixel_values.append(pixel_values)
        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
            assert label_0 in (0, 1)
            if label_0==0: # flip
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0) # no batch dim
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def preprocess_data_sft(examples, transforms):
        images = []
        # Probably cleaner way to do this iteration
        for im_0_bytes, im_1_bytes, label_0 in zip(examples['jpg_0'], examples['jpg_1'], examples['label_0']):
            assert label_0 in (0, 1)
            im_bytes = im_0_bytes if label_0==1 else im_1_bytes
            images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
        examples["pixel_values"] = [transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return_d =  {"pixel_values": pixel_values}
        return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
        return_d["user_ids"] = torch.tensor([example['user_id'] for example in examples])
        return return_d

    if args.train_method == 'dpo':
        def preprocess_train(examples):
            return preprocess_data_dpo(examples, train_transforms)
    else:
        def preprocess_train(examples):
            return preprocess_data_sft(examples, train_transforms)

    def preprocess_valid(examples):
        return preprocess_data_dpo(examples, valid_transforms)
    #### END PREPROCESSING/COLLATION ####

    def remove_split_label(dataset):
        # eliminate no-decisions (0.5-0.5 labels)
        orig_len = dataset.num_rows
        not_split_idx = [i for i, label_0 in enumerate(dataset['label_0']) if label_0 in (0, 1)]
        dataset = dataset.select(not_split_idx)
        new_len = dataset.num_rows
        logger.info(f"Eliminated {orig_len - new_len}/{orig_len} in Pick-a-pic")
        return dataset

    def sample_from_list(l, n):
        # sample n items from l, return remaining items too
        indices = list(range(len(l)))
        yes_indices = random.sample(indices, n)
        no_indices = [i for i in indices if i not in set(yes_indices)]
        return [l[i] for i in yes_indices], [l[i] for i in no_indices]

    ### DATASET #####
    with accelerator.main_process_first():
        dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        dataset = remove_split_label(dataset)
        # subset users for debug/mini runs
        unique_user_ids = list(set(dataset['user_id']))
        select_user_ids = random.sample(unique_user_ids, math.ceil(len(unique_user_ids) * args.max_user_proportion))
        select_ids = [i for i, u in enumerate(dataset['user_id']) if u in set(select_user_ids)]
        dataset = dataset.select(select_ids)
        # split into known user train and valid
        user_id_counts = Counter(dataset['user_id'])
        known_user_to_ids = defaultdict(list)
        for i, u in enumerate(dataset['user_id']):
            if user_id_counts[u] >= args.per_user_contribution_thr:
                known_user_to_ids[u].append(i)
        assert all(len(v) >= args.per_user_contribution_thr for v in known_user_to_ids.values())
        known_train_ids, known_val_ids = [], []
        for _, ids in known_user_to_ids.items():
            x, y = sample_from_list(ids, args.per_user_n_eval)
            known_train_ids += y
            known_val_ids += x
        # get unknown user valid
        unknown_ids = [i for i, u in enumerate(dataset['user_id']) if user_id_counts[u] < args.per_user_contribution_thr]
        # build dataset
        known_train_dataset = dataset.select(known_train_ids).shuffle(seed=args.seed)
        known_valid_dataset = dataset.select(known_val_ids).shuffle(seed=args.seed)
        unknown_valid_dataset = dataset.select(unknown_ids).shuffle(seed=args.seed)
        logger.info(f'Known train dataset size: {len(known_train_dataset)}')
        logger.info(f'Known valid dataset size: {len(known_valid_dataset)}')
        logger.info(f'Unknown valid dataset size: {len(unknown_valid_dataset)}')
        assert len(known_train_dataset) + len(known_valid_dataset) + len(unknown_valid_dataset) == len(dataset)
        # transform
        known_train_dataset_transformed = known_train_dataset.with_transform(preprocess_train)
        known_valid_dataset_transformed = known_valid_dataset.with_transform(preprocess_valid)
        unknown_valid_dataset_transformed = unknown_valid_dataset.with_transform(preprocess_valid)

    # get the 500 unique validation captions for evaluation
    all_known_valid_captions = sorted(set(known_valid_dataset['caption'])) # deterministic
    all_unknown_valid_captions = sorted(set(unknown_valid_dataset['caption'])) # deterministic
    random.shuffle(all_known_valid_captions)
    random.shuffle(all_unknown_valid_captions)
    if args.max_valid_captions is not None:
        all_known_valid_captions = all_known_valid_captions[:args.max_valid_captions]
    if args.max_valid_captions is not None:
        all_unknown_valid_captions = all_unknown_valid_captions[:args.max_valid_captions]
    # save validation captions
    with open(os.path.join(args.output_dir, 'all_known_valid_captions.txt'), 'w') as f:
        f.write('\n'.join(all_known_valid_captions))
    with open(os.path.join(args.output_dir, 'all_unknown_valid_captions.txt'), 'w') as f:
        f.write('\n'.join(all_unknown_valid_captions))

    # DataLoaders creation:
    known_train_dataloader = torch.utils.data.DataLoader(
        known_train_dataset_transformed,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    known_valid_dataloader = torch.utils.data.DataLoader(
        known_valid_dataset_transformed,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )
    unknown_valid_dataloader = torch.utils.data.DataLoader(
        unknown_valid_dataset_transformed,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=False
    )
    ##### END BIG OLD DATASET BLOCK #####

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    #### START ACCELERATOR PREP ####
    vae, unet, text_encoder, ref_unet, optimizer, known_train_dataloader, known_valid_dataloader, unknown_valid_dataloader, lr_scheduler = accelerator.prepare(
        vae, unet, text_encoder, ref_unet, optimizer, known_train_dataloader, known_valid_dataloader, unknown_valid_dataloader, lr_scheduler
    )
    # for x in [vae, unet, text_encoder, ref_unet]: logger.info(x.device)
    # for x in [vae, unet, text_encoder, ref_unet]: logger.info(x.dtype)
    # for p in vae.parameters(): assert not p.requires_grad
    # for p in text_encoder.parameters(): assert not p.requires_grad
    # for p in unet.parameters(): assert p.requires_grad
    # for p in ref_unet.parameters(): assert not p.requires_grad
    ### END ACCELERATOR PREP ###

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(known_train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={"wandb": {"name": args.tag}}
        )

    # validation utils
    @torch.no_grad()
    def generate_images(unet, captions, batch_size):
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        diffusers.utils.logging.set_verbosity_error()
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet), # PNDMScheduler
        ).to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = None # disable nsfw

        imgs = []
        with torch.autocast("cuda"):
            n_batches = math.ceil(len(captions) / batch_size)
            for caps in tqdm(chunks(captions, batch_size), total=n_batches):
                imgs = pipeline(caps, num_inference_steps=args.num_inference_steps, generator=generator).images
        diffusers.utils.logging.set_verbosity_warning()
        del pipeline
        return imgs

    @torch.no_grad()
    def compute_median_pickscore_reward(imgs, captions, batch_size):
        all_imgs_n_captions = list(zip(imgs, captions))
        all_scores = []
        with torch.autocast("cuda"):
            n_batches = math.ceil(len(all_imgs_n_captions) / batch_size)
            for imgs_n_captions in tqdm(chunks(all_imgs_n_captions, batch_size), total=n_batches):
                imgs, caps = [x[0] for x in imgs_n_captions], [x[1] for x in imgs_n_captions]
                all_scores += pickscore_model.score(imgs, caps)
        return torch.median(torch.tensor(all_scores)).item()

    @torch.no_grad()
    def compute_implicit_acc(unet, ref_unet, dataloader):
        avg_implicit_acc = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
            latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(batch["input_ids"])[0].repeat(2, 1, 1)
            noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)\
                .long().chunk(2)[0].repeat(2)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            assert noise_scheduler.config.prediction_type == "epsilon"
            target = noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            _, _, _, implicit_acc = compute_loss_dpo(args, model_pred, target, ref_unet, encoder_hidden_states, noisy_latents, timesteps)
            avg_implicit_acc += implicit_acc * batch["input_ids"].shape[0]
        return avg_implicit_acc / len(dataloader.dataset)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    n_trainable_params = count_param_in_model(unet, trainable_only=True)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(known_train_dataset_transformed)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total number of parameters to train = {n_trainable_params}") # 797184 for lora4, 3188736 for lora16, 859520964 for unet

    global_step = 0

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    #### START MAIN TRAINING LOOP #####
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(known_train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == 'dpo':
                    # y_w and y_l were concatenated along channel dimension
                    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
                    # If using AIF then we haven't ranked yet so do so now
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]

                #### Diffusion Stuff ####
                # encode pixels --> latents
                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                if args.train_method == 'dpo': # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                ### START PREP BATCH ###
                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                if args.train_method == 'dpo':
                    encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####

                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise

                # Make the prediction from the model we're learning
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = compute_loss_sft(model_pred, target)
                elif args.train_method == 'dpo':
                    loss, raw_model_loss, raw_ref_loss, implicit_acc = compute_loss_dpo(
                        args, model_pred, target, ref_unet, encoder_hidden_states, noisy_latents, timesteps
                    )
                #### END LOSS COMPUTATION ###

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == 'dpo':
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        lora_layers if args.lora_rank > 0 else unet.parameters(),
                        args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method == 'dpo':
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # save model
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(unet, save_path)
                        logger.info(f"Saved state to {save_path}")

                        # make validation dir
                        val_dir = os.path.join(args.output_dir, f'eval_{global_step}')
                        os.makedirs(val_dir, exist_ok=True)

                        # validation
                        # part 1: generate some images from unet and ref unet, upload few to wandb
                        logger.info('generating known valid images from unet')
                        unet_known_valid_imgs = generate_images(unet, all_known_valid_captions, args.valid_batch_size)
                        logger.info('generating unknown valid images from unet')
                        unet_unknown_valid_imgs = generate_images(unet, all_unknown_valid_captions, args.valid_batch_size)
                        for i, img in enumerate(unet_known_valid_imgs): img.save(os.path.join(val_dir, f'unet_known_{i}.jpg'))
                        for i, img in enumerate(unet_unknown_valid_imgs): img.save(os.path.join(val_dir, f'unet_unknown_{i}.jpg'))
                        accelerator.log({
                            "unet_known_valid_images": [wandb.Image(img, caption=f"{i}: {all_known_valid_captions[i]}") for i, img in enumerate(unet_known_valid_imgs[:args.n_log_valid_imgs])],
                            "unet_unknown_valid_images": [wandb.Image(img, caption=f"{i}: {all_unknown_valid_captions[i]}") for i, img in enumerate(unet_unknown_valid_imgs[:args.n_log_valid_imgs])],
                        }, step=global_step)

                        # if first time, log ref unet performance
                        if global_step == args.checkpointing_steps:
                            logger.info('generating known valid images from ref')
                            ref_known_valid_imgs = generate_images(ref_unet, all_known_valid_captions, args.valid_batch_size)
                            logger.info('generating unknown valid images from ref')
                            ref_unknown_valid_imgs = generate_images(ref_unet, all_unknown_valid_captions, args.valid_batch_size)
                            for i, img in enumerate(ref_known_valid_imgs): img.save(os.path.join(val_dir, f'ref_known_{i}.jpg'))
                            for i, img in enumerate(ref_unknown_valid_imgs): img.save(os.path.join(val_dir, f'ref_unknown_{i}.jpg'))
                            accelerator.log({
                                "ref_known_valid_images": [wandb.Image(img, caption=f"{i}: {all_known_valid_captions[i]}") for i, img in enumerate(ref_known_valid_imgs[:args.n_log_valid_imgs])],
                                "ref_unknown_valid_images": [wandb.Image(img, caption=f"{i}: {all_unknown_valid_captions[i]}") for i, img in enumerate(ref_unknown_valid_imgs[:args.n_log_valid_imgs])],
                            }, step=global_step)

                        # part 2: median pickscore reward
                        logger.info('computing known median pickscore from unet')
                        unet_known_median_pickscore = compute_median_pickscore_reward(unet_known_valid_imgs, all_known_valid_captions, args.valid_batch_size)
                        logger.info('computing unknown median pickscore from unet')
                        unet_unknown_median_pickscore = compute_median_pickscore_reward(unet_unknown_valid_imgs, all_unknown_valid_captions, args.valid_batch_size)
                        with open(os.path.join(val_dir, 'unet_pickscore.txt'), 'w') as f:
                            f.write(f'known: {str(unet_known_median_pickscore)}\nunknown: {str(unet_unknown_median_pickscore)}')
                        accelerator.log({
                            "unet_known_median_pickscore": unet_known_median_pickscore,
                            "unet_unknown_median_pickscore": unet_unknown_median_pickscore,
                        }, step=global_step)

                        # if first time, log ref unet performance
                        if global_step == args.checkpointing_steps:
                            logger.info('computing known median pickscore from unet')
                            ref_known_median_pickscore = compute_median_pickscore_reward(ref_known_valid_imgs, all_known_valid_captions, args.valid_batch_size)
                            logger.info('computing unknown median pickscore from unet')
                            ref_unknown_median_pickscore = compute_median_pickscore_reward(ref_unknown_valid_imgs, all_unknown_valid_captions, args.valid_batch_size)
                            with open(os.path.join(val_dir, 'ref_pickscore.txt'), 'w') as f:
                                f.write(f'known: {str(ref_known_median_pickscore)}\nunknown: {str(ref_unknown_median_pickscore)}')
                            accelerator.log({
                                "ref_known_median_pickscore": ref_known_median_pickscore,
                                "ref_unknown_median_pickscore": ref_unknown_median_pickscore,
                            }, step=global_step)

                        # part 3: estimate implicit acc with 10 random timesteps for pickapic validation set
                        logger.info('computing known implicit accuracy from unet')
                        unet_known_implicit_acc = sum(compute_implicit_acc(unet, ref_unet, known_valid_dataloader) for _ in range(10)) / 10
                        logger.info('computing unknown implicit accuracy from unet')
                        unet_unknown_implicit_acc = sum(compute_implicit_acc(unet, ref_unet, unknown_valid_dataloader) for _ in range(10)) / 10
                        with open(os.path.join(val_dir, 'unet_implicit_acc.txt'), 'w') as f:
                            f.write(f'known: {str(unet_known_implicit_acc)}\nunknown: {str(unet_unknown_implicit_acc)}')
                        accelerator.log({
                            "unet_known_implicit_acc": unet_known_implicit_acc,
                            "unet_unknown_implicit_acc": unet_unknown_implicit_acc,
                        }, step=global_step)

                        # TODO part 4: CLIP text-to-image alignment
                        # TODO part 5: FID and stuff to see if KL-divergence is working

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save model
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(save_path):
            save_model(unet, save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
