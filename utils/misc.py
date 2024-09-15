from typing import List, Union
import torch
import torch.nn.functional as F


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def count_param_in_model(model, trainable_only=True):
    count = 0
    for p in model.parameters():
        if not trainable_only or p.requires_grad:
            count += torch.numel(p)
    return count

def compute_loss_sft(model_pred, target):
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def compute_loss_dpo(args, model_pred, target, ref_unet, encoder_hidden_states, noisy_latents, timesteps):
    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
    # losses are both 2 * LBS
    # 1st half of tensors is preferred (y_w), second half is unpreferred
    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
    model_losses_w, model_losses_l = model_losses.chunk(2) # SEθ_W, SEθ_L both are (LBS,)
    model_diff = model_losses_w - model_losses_l # SEθ_W - SEθ_L

    with torch.no_grad(): # Get the reference policy (unet) prediction
        ref_pred = ref_unet(noisy_latents, timesteps, encoder_hidden_states).sample.detach()
        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
        ref_losses_w, ref_losses_l = ref_losses.chunk(2) # SEref_W, SEref_L both are (LBS,)
        ref_diff = ref_losses_w - ref_losses_l # SEref_W - SEref_L

    scale_term = -0.5 * args.beta_dpo
    model_ref_diff = model_diff - ref_diff # (SEθ_W - SEref_W) - (SEθ_L - SEref_L)
    inside_term = scale_term * model_ref_diff
    loss = -1 * F.logsigmoid(inside_term).mean()

    # log raw model loss and implicit acc
    raw_model_loss = model_losses.mean()
    raw_ref_loss = ref_losses.mean()
    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    return loss, raw_model_loss, raw_ref_loss, implicit_acc
