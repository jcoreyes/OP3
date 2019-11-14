import numpy as np
import torch
from torchvision.utils import save_image
from op3.torch import pytorch_util as ptu
import pdb

#images (W,H,3,D,D), values should be between 0 and 1
def create_image_from_subimages(images, file_name):
    cur_shape = images.shape
    images = images.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
    images = images.view(-1, *cur_shape[-3:])  # (H*W, 3, D, D)
    save_image(images, filename=file_name, nrow=cur_shape[0])
    #Note: Weird that it is cur_shape[0] but cur_shape[1] produces incorrect image


# Inputs: true_images (T1,3,D,D),  colors (T,K,3,D,D),  masks (T,K,1,D,D), schedule (T) np
#   file_name (string),  quicksave_type is either "full" or "subimages"
# Images are torch tensors, schedule is numpy array
def quicksave(true_images, colors, masks, schedule, file_name, quicksave_type):
    if np.max(schedule) == 2:
        schedule = schedule//2  # We want schedule to be 0's and 1's
    recons = (colors * masks).sum(1) #(T,3,D,D)

    #If we are doing rollouts (i.e. T1 >= sum(schedule))
    true_images = torch.cat([true_images, torch.zeros_like(true_images[:1].to(true_images.device))], dim=0)
    tmp = np.where(np.cumsum(schedule) < true_images.shape[0] - 1, np.cumsum(schedule), -1)
    true_images = true_images[tmp]
    # true_images = true_images[min(np.cumsum(schedule), true_images.shape[0]-1)] #(T,3,D,D) #NOTE: This only works for schedules with 0's and 1's!!

    # true_images = torch.where(np.cumsum(schedule) < true_images.shape[0], true_images, ptu.zeros_like(true_images))

    full_plot = torch.cat([true_images.unsqueeze(1), recons.unsqueeze(1)], dim=1) #(T,2,3,D,D)
    if quicksave_type == "full":
        subimages = colors * masks #(T,K,3,D,D)
        masks = masks.repeat(1,1,3,1,1) #(T,K,3,D,D)
        full_plot = torch.cat([full_plot, masks, subimages], dim=1) #(T,2+K+K,3,D,D)
    elif quicksave_type == "subimages":
        subimages = colors * masks #(T,K,3,D,D)
        full_plot = torch.cat([full_plot, subimages], dim=1)  # (T,2+K,3,D,D)
    else:
        raise ValueError("Invalid value '{}' given to quicksave".format(quicksave_type))
    create_image_from_subimages(full_plot, file_name)


# Input: state_info_dict with B=1, file_name (Str), true_image (B=1,1,3,D,D) or (1,3,D,D) or (3,D,D)
# Output: N/A. Visualizes subimages, mask, reconstruction
def visualize_state_info(state_info_dict, file_name, true_image=None):
    sub_images = state_info_dict["sub_images"]  # (B,K,3,D,D)
    masks = state_info_dict["masks"].repeat(1,1,3,1,1)  # (B,K,1,D,D) -> (B,K,3,D,D)
    final_recon = state_info_dict["final_recon"]  # (B,3,D,D)
    colors = state_info_dict["colors"]  # (B,K,3,D,D)

    # pdb.set_trace()

    if true_image is not None:
        if len(true_image.shape) == 3:
            true_image = true_image.unsqueeze(0).unsqueeze(0) # (B=1,1,3,D,D)
        elif len(true_image.shape) == 4:
            true_image = true_image.unsqueeze(0)  # (B=1,1,3,D,D)
        # true_image = true_image.unsqueeze(0).unsqueeze(0)  # (B=1,1,3,D,D)
        full_plot = torch.cat([true_image, final_recon.unsqueeze(1), sub_images, masks, colors], dim=1)  # (B=1,1+1+K+K+k,3,D,D)
    else:
        full_plot = torch.cat([final_recon.unsqueeze(1), sub_images, masks, colors], dim=1)  # (B=1,1+K+K+k,3,D,D)

    create_image_from_subimages(full_plot, file_name)  # full_plot is (1,?,3,D,D)


    # important_info["colors"] = torch.cat(important_info["colors"])  # (B,K,3,D,D)
    # important_info["masks"] = torch.cat(important_info["masks"])  # (B,K,1,D,D)
    # important_info["sub_images"] = torch.cat(important_info["sub_images"])  # (B,K,3,D,D)
    # important_info["final_recon"] = torch.cat(important_info["final_recon"])  # (B,3,D,D)
    # important_info["state"] = self._stack_state(important_info["state"])  # State with (B,*) entries



