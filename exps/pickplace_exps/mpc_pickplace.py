import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # Note, this is needed for 3d plotting even if it doesn't seem used!
import matplotlib.cm as cm
from matplotlib import colors

# import sys
# sys.path.append("../../")

from op3.torch import pytorch_util as ptu
import op3.torch.op3_modules.visualizer as visualizer
from op3.util.plot import plot_multi_image
from torchvision.utils import save_image
import torch
import numpy as np
from argparse import ArgumentParser
from op3.launchers.launcher_util import run_experiment

import op3.torch.op3_modules.op3_model as op3_model
from op3.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from op3.util.misc import get_module_path
from exps.pickplace_exps.saved_models.model_parameters_info import params_to_info

from collections import OrderedDict, defaultdict
import shutil
import pickle
import json
import imageio
import os
import pdb

##############Cost Class ############
class Cost:
    def __init__(self, logging_directory, core_type="subimage", compare_func='mse', post_process='raw', aggregate='sum'):
        self.logging_directory = logging_directory
        self.core_type = core_type
        if core_type not in ["subimage", "final_recon", "latent"]:
            raise ValueError("Invalid value of core_type: {}".format(core_type))
        self.compare_func = compare_func
        self.post_process = post_process
        self.aggregate = aggregate


    # Inputs: goal_latents (n_goal_latents=K,R), goal_latents_recon (n_goal_latents=K,3,64,64)
    # goal_image (1,3,64,64), pred_latents (n_actions,K,R), pred_latents_recon (n_actions,K,3,64,64)
    # pred_images (n_actions,3,64,64)
    def get_action_rankings(self, goal_latents, goal_latents_recon, goal_image, pred_latents,
                    pred_latents_recon, pred_image, image_suffix="", plot_actions=8):
        self.image_suffix = image_suffix
        self.plot_actions = plot_actions

        if self.core_type == "final_recon":  # Pretend that we only did K=1
            goal_latents_recon = goal_image  # (K=1,3,64,64)
            pred_latents_recon = pred_image.unsqueeze(1)  # (n_actions,K=1,3,64,64)

        if self.aggregate == 'sum':
            return self.sum_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        elif self.aggregate == 'min':
            return self.min_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        else:
            raise KeyError

    def get_single_costs(self, goal_latent, goal_latent_recon, pred_latent, pred_latent_recon):
        if self.core_type == 'subimage' or self.core_type == "final_recon":
            dists = self.compare_subimages(goal_latent_recon, pred_latent_recon)
        elif self.core_type == 'latent':
            dists = self.compare_latents(goal_latent, pred_latent)
        else:
            raise ValueError("Invalid core_type: {}".format(self.core_type))

        costs = self.post_process_func(dists)
        return costs


    def compare_latents(self, goal_latent, pred_latent):
        if self.compare_func == 'mse':
            return self.mse(goal_latent.view(1, 1, -1), pred_latent)
        else:
            raise KeyError

    # Input: goal_latent_recon (3,D,D),  pred_latent_recon (Na,K,3,D,D), where Na is the number of actions
    def compare_subimages(self, goal_latent_recon, pred_latent_recon):
        if self.compare_func == 'mse':
            return self.image_mse(goal_latent_recon.view((1, 1, 3, 64, 64)), pred_latent_recon)
        elif self.compare_func == 'psuedo_intersect':
            # pdb.set_trace()
            m1 = (goal_latent_recon > 0.01).float()  # (3,D,D)
            m2 = (pred_latent_recon > 0.01).float()  # (Na,K,3,D,D)
            intersect = (m1 * m2).float()  # (Na,K,3,D,D)
            intersect.sum((-3, -2, -1))    # (Na,K)
            union = m1.sum((-3, -2, -1)) + m2.sum((-3, -2, -1))  # (Na,K)
            threshold_intersect = (intersect * (torch.pow(goal_latent_recon - pred_latent_recon, 2) < 0.08).float()).sum((-3, -2, -1))  # (Na,K)
            iou = threshold_intersect / union  # (Na,K)
            return 1 - iou
        else:
            raise KeyError


    def post_process_func(self, dists):
        if self.post_process == 'raw':
            return dists
        elif self.post_process == 'negative_exp':
            return -torch.exp(-dists)
        else:
            raise KeyError

    # Inputs: goal_latents (n_goal_latents=K,R), goal_latents_recon (n_goal_latents=K,3,64,64)
    # goal_image (1,3,64,64), pred_latents (n_actions,K,R), pred_latents_recon (n_actions,K,3,64,64)
    # pred_images (n_actions,3,64,64)
    def sum_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_images):

        n_goal_latents = goal_latents_recon.shape[0] #Note, this should equal K if we did not filter anything out
        # Compare against each goal latent
        costs = []  #(n_goal_latents, n_actions)
        latent_idxs = [] # (n_goal_latents, n_actions), [a,b] is an index corresponding to a latent
        for i in range(n_goal_latents): #Going through all n_goal_latents goal latents
            # pdb.set_trace()
            single_costs = self.get_single_costs(goal_latents[i], goal_latents_recon[i], pred_latents, pred_latents_recon)
            min_costs, latent_idx = single_costs.min(-1)  # take min among K, size is (n_actions)

            costs.append(min_costs)
            latent_idxs.append(latent_idx)

        costs = torch.stack(costs) # (n_goal_latents, n_actions)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions)

        #Sort by sum cost
        #Image contains the following: Pred_images, goal_latent_reconstructions, and
        # corresponding pred_latent_reconstructions
        #For every latent in goal latents, find corresponding predicted one (this is in latent_idxs)
        #  Should have something that is (K, num_actions) -> x[a,b] is index for pred_latents_recon
        sorted_costs, best_action_idxs = costs.sum(0).sort()

        if self.plot_actions:
            sorted_pred_images = pred_images[best_action_idxs]

            corresponding_pred_latent_recons = []
            for i in range(n_goal_latents):
                tmp = pred_latents_recon[best_action_idxs, latent_idxs[i, best_action_idxs]]  # (n_actions, 3, 64, 64)
                corresponding_pred_latent_recons.append(tmp)
            corresponding_pred_latent_recons = torch.stack(corresponding_pred_latent_recons)  # (n_goal_latents, n_actions, 3, 64, 64)
            corresponding_costs = costs[:, best_action_idxs]

            full_plot = torch.cat([sorted_pred_images.unsqueeze(0),  # (1, n_actions, 3, 64, 64)
                                   corresponding_pred_latent_recons,  # (n_goal_latents, n_actions, 3, 64, 64)
                                   ], 0)
            plot_size = self.plot_actions
            full_plot = full_plot[:, :plot_size]

            #Add goal latents
            tmp = torch.cat([goal_image, goal_latents_recon], dim=0).unsqueeze(1)  # (n_goal_latents+1, 1, 3, 64, 64)
            full_plot = torch.cat([tmp, full_plot], dim=1)

            #Add captions
            caption = np.zeros(full_plot.shape[:2])
            caption[0, 1:] = ptu.get_numpy(sorted_costs[:plot_size])
            caption[1:1+n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:plot_size])[:,:plot_size]

            plot_multi_image(ptu.get_numpy(full_plot),
                             '{}/{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)

        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), np.zeros(len(sorted_costs))

    def min_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image):

        n_goal_latents = goal_latents_recon.shape[0]
        # Compare against each goal latent
        costs = []  # (n_goal_latents, n_actions)
        latent_idxs = []  # (n_goal_latents, n_actions), [a,b] is an index corresponding to a latent
        for i in range(n_goal_latents):  # Going through all n_goal_latents goal latents
            single_costs = self.get_single_costs(goal_latents[i], goal_latents_recon[i], pred_latents, pred_latents_recon) #(K, n_actions)
            min_costs, latent_idx = single_costs.min(-1)  # take min among K, size is (n_actions)

            costs.append(min_costs)
            latent_idxs.append(latent_idx)

        costs = torch.stack(costs) # (n_goal_latents, n_actions)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions)

        #Sort by sum cost
        #Image contains the following: Pred_images, goal_latent_reconstructions, and
        # corresponding pred_latent_reconstructions
        #For every latent in goal latents, find corresponding predicted one (this is in latent_idxs)
        #  Should have something that is (K, num_actions) -> x[a,b] is index for pred_latents_recon
        min_costs, min_goal_latent_idx = costs.min(0) #(num_actions)
        sorted_costs, best_action_idxs = min_costs.sort() #(num_actions)

        if self.plot_actions:
            sorted_pred_images = pred_image[best_action_idxs]
            corresponding_pred_latent_recons = []
            for i in range(n_goal_latents):
                tmp = pred_latents_recon[best_action_idxs, latent_idxs[i, best_action_idxs]] #(n_actions, 3, 64, 64)
                corresponding_pred_latent_recons.append(tmp)
            corresponding_pred_latent_recons = torch.stack(corresponding_pred_latent_recons) #(n_goal_latents, n_actions, 3, 64, 64)
            corresponding_costs = costs[:, best_action_idxs] # (n_goal_latents, n_actions)

            # pdb.set_trace()
            min_corresponding_latent_recon = pred_latents_recon[best_action_idxs, latent_idxs[min_goal_latent_idx[best_action_idxs], best_action_idxs]] #(n_actions, 3, 64, 64)

            # pdb.set_trace()

            full_plot = torch.cat([sorted_pred_images.unsqueeze(0), # (1, n_actions, 3, 64, 64)
                                   corresponding_pred_latent_recons, # (n_goal_latents=K, n_actions, 3, 64, 64)
                                   min_corresponding_latent_recon.unsqueeze(0) #(1, n_actions, 3, 64, 64)
                                   ], 0) # (n_goal_latents+2, n_actions, 3, 64, 64)
            plot_size = self.plot_actions
            full_plot = full_plot[:, :plot_size] # (n_goal_latents+2, plot_size, 3, 64, 64)

            # Add goal latents
            tmp = torch.cat([goal_image, goal_latents_recon, goal_image], dim=0).unsqueeze(1)  # (n_goal_latents+2, 1, 3, 64, 64)
            full_plot = torch.cat([tmp, full_plot], dim=1) # (n_goal_latents+2, plot_size+1, 3, 64, 64)

            #Add captions
            caption = np.zeros(full_plot.shape[:2])
            caption[0, 1:] = ptu.get_numpy(sorted_costs[:plot_size])
            caption[1:1 + n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:, :plot_size])

            plot_multi_image(ptu.get_numpy(full_plot),
                             '{}/mpc_pred_{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)


        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), ptu.get_numpy(min_goal_latent_idx)

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def image_mse(self, im1, im2):
        # im1, im2 are (*, 3, D, D)
        # Note: * dimensions may not be equal between im1, im2 so automatically broadcast over them
        return torch.pow(im1 - im2, 2).mean((-1, -2, -3))  # Takes means across the last dimensions (3, D, D)


#############Start Action Selection Class#########
####Stage3 Specific
class Stage3_CEM:
    def __init__(self, logging_dir, cem_steps, num_samples, time_horizon, score_actions_class, action_type=None):
        self.logging_dir = logging_dir
        self.cem_steps = cem_steps
        self.num_samples = num_samples
        self.time_horizon = time_horizon
        self.score_actions_class = score_actions_class
        self.action_type = action_type  # action_type is None or an integer representing K
        self.env = None
        self.goal_info = None
        self.num_loc_samples = 500

    def select_action(self, goal_info, initial_hidden_state, env, model, logging_suffix):
        self.env = env
        self.goal_info = goal_info
        self.initial_hidden_state = initial_hidden_state

        # pdb.set_trace()
        if self.action_type is not None:
            self._get_latent_locs(model, self.initial_hidden_state, file_name=logging_suffix)

        return self._cem(model, logging_suffix)

    # Input: hidden_state with B=1
    # Output: pick locations per each hidden state (K,2)
    def _get_latent_locs(self, model, hidden_state, file_name=None):
        rand_actions = ptu.from_numpy(np.stack([self.env.sample_action() for _ in range(self.num_loc_samples)])) # (B,A)
        state_action_attention, interaction_attention, all_delta_vals, all_lambdas_deltas = \
            model.get_all_activation_values(hidden_state, rand_actions)

        interaction_attention = interaction_attention.sum(-1)  # (B,K,K-1) -> (B,K)
        normalized_weights = interaction_attention / interaction_attention.sum(0)  #(B,K)
        mean_point = (normalized_weights.unsqueeze(2) * rand_actions[:, :2].unsqueeze(1))  # ((B,K)->(B,K,1) * (B,2)->(B,1,2)) -> (B,K,2)
        mean_point = mean_point.sum(0)  #(B,K,2) -> (K,2)

        if file_name is not None:
            plot_action_vals(self.env, ptu.get_numpy(interaction_attention), ptu.get_numpy(rand_actions),
                             "{}/{}_pick_locs".format(self.logging_dir, file_name), is_normalized=True)

        return mean_point

    # Input: latent actions (B=Na,T,A=Nl+2), initial_hidden_state (B=1), all pytorch tensors
    # Output: predicted_info
    def _latent_batch_internal_inference(self, model, actions, initial_hidden_state):
        # Replicate initial_hidden_state_and_info to make B=1 to B=Na
        # For t = 1 to T:
            # For each hidden state, we need to get a pick location pick_locs (K,2)
            # Once we have pick locations for all hidden states, we can create the appriopriate env actions (B,A=raw)
            # Apply model.batch_internal_inference() with proper actions
            # Get new hidden states, repeat

        num_actions = actions.shape[0]
        all_hidden_states = model.replicate_state(initial_hidden_state, num_actions)  # (B=Na)
        all_pick_locs = ptu.zeros(num_actions, self.time_horizon, self.action_type, 2)  # (B,T,K,2)
        all_env_actions = ptu.zeros(num_actions, self.time_horizon, 4)  # (B,T,4)
        schedule = np.array([1])  # We only want to do a rollout of just a single action
        for t in range(self.time_horizon):
            for i in range(num_actions):
                single_state = model.select_specific_state(all_hidden_states, [i])
                pick_locs = self._get_latent_locs(model, single_state)  # (K,2)
                if t == 0:
                    all_pick_locs[:, t] = pick_locs  # Broadcast across everything as we have the same initial state
                    break
                else:
                    all_pick_locs[i, t] = pick_locs

            raw_actions = actions[:, t, :-2]  # (B,K), note these are one hot vectors
            selected_pick_locs = (raw_actions.unsqueeze(2) * all_pick_locs[:, t]).sum(1)  # (B,K,1)*(K,2)->(B,K,2)->(B,2)  Note that we use the one hot encoding as a mask
            env_actions = torch.cat([selected_pick_locs, actions[:, t, -2:]], dim=1)  # (B,2) (B,2) -> (B,4)
            all_env_actions[:, t] = env_actions  # (B,4)

            env_actions = env_actions.unsqueeze(1)  # (B,1,4)
            predicted_info = model.batch_internal_inference(obs=None, actions=env_actions,
                                                            initial_hidden_state=all_hidden_states,
                                                            schedule=schedule, figure_path=None)
            all_hidden_states = predicted_info["state"]
        return predicted_info, all_env_actions

    # Inputs: actions (B,T,A) np,  model
    # Outputs: Index of actions based off sorted costs (B), paired goal latent (for removal) (B), final_recons (B,3,D,D)
    def _random_shooting(self, actions, model, image_suffix):
        # Like internal_inference except initial_hidden_state might only contain one state while obs/actions contain (B,*)
        # Inputs: obs (B,T1,3,D,D) or None, actions (B,T2,A) or None, initial_hidden_state or None, schedule (T3)
        #   Note: Assume that initial_hidden_state has entries of size (B=1,*)

        goal_info = self.goal_info
        schedule = np.array([1]*actions.shape[1])
        actions = ptu.from_numpy(actions)
        if self.action_type is None:
            predicted_info = model.batch_internal_inference(obs=None, actions=actions, initial_hidden_state=self.initial_hidden_state,
                                                            schedule=schedule, figure_path=None)
            all_env_actions = actions
        else:
            predicted_info, all_env_actions = self._latent_batch_internal_inference(model, actions, self.initial_hidden_state)

        # Inputs to get_action_rankings(): goal_latents (n_goal_latents=K,R),
        # goal_latents_recon (n_goal_latents=K,3,64,64), goal_image (1,3,64,64), pred_latents (n_actions,K,R),
        # pred_latents_recon (n_actions,K,3,64,64),  pred_images (n_actions,3,64,64)
        sorted_costs, best_actions_indices, goal_latent_indices = self.score_actions_class.get_action_rankings(
            goal_info["state"]["post"]["samples"][0], goal_info["sub_images"][0], goal_info["goal_image"],
            predicted_info["state"]["post"]["samples"], predicted_info["sub_images"], predicted_info["final_recon"],
            image_suffix = image_suffix)

        num_plot_actions = 20
        self.plot_action_errors(self.env, all_env_actions[best_actions_indices][:num_plot_actions, 0],
                                predicted_info["final_recon"][best_actions_indices][:num_plot_actions],
                                image_suffix+"_action_errors")


        best_single_env_action = all_env_actions[best_actions_indices[0]]

        return best_actions_indices, goal_latent_indices, predicted_info["final_recon"], ptu.get_numpy(best_single_env_action)

    # Inputs: N/A
    # Outputs: random actions (B,T,A)
    def _get_initial_actions(self):
        actions = []
        for i in range(self.time_horizon):
            if self.action_type is None:  # Normal action space
                actions.append(np.stack([self.env.sample_action() for _ in range(self.num_samples)]))  # (B,A)
            else:  # Object action space
                tmp = np.stack([self.env.sample_action() for _ in range(self.num_samples)])  # (B,A=4)
                cur_actions = np.zeros((self.num_samples, self.action_type + 2))
                cur_actions[:, -2:] = tmp[:, -2:]

                #https://stackoverflow.com/questions/45093615/random-one-hot-matrix-in-numpy
                rand_latents_idxs = np.eye(self.action_type)[np.random.choice(self.action_type, self.num_samples)]  # (B, Nl)
                cur_actions[:, :-2] = rand_latents_idxs
                actions.append(cur_actions)

        actions = np.array(actions).transpose((1, 0, 2))  # (B,T,A)
        return np.array(actions)

    # Input: model, logging_suffix (Str)
    # Output: best actions (T,A), corresponding latent (Sc), corresponding predicted reconstructions (3,D,D)
    def _cem(self, model, logging_suffix):
        actions = self._get_initial_actions()  # (B,T,A)
        filter_cutoff = int(self.num_samples * 0.1)  # F

        for i in range(self.cem_steps):
            best_actions_indices, goal_latent_indices, pred_recons, best_single_env_action \
                = self._random_shooting(actions, model, "action_{}_{}".format(logging_suffix, i))  # all (B), except for best_single_env_action (A=4)
            sorted_actions = actions[best_actions_indices]  # (B,T,A)
            sorted_best_actions = sorted_actions[:filter_cutoff]  # (F,T,A)
            actions = self._resample_action(sorted_best_actions)  # (B,T,A)
            print("CEM step: {}".format(i))

        best_action_index = best_actions_indices[0]
        return np.array(best_single_env_action), goal_latent_indices[best_action_index], pred_recons[best_action_index]

    # Input: best_actions (B,T,A)
    # Output: New actions sampled for distributions fit on the best actions
    def _resample_action(self, best_actions):
        if self.action_type == None:
            mean = best_actions.mean(0)  # (T,A)
            std = best_actions.std(0) + 0.05  # (T,A), +0.05 is added as a minimum std deviation (note pick threshold is 0.2)
            actions = self.env.sample_multiple_action_gaussian(mean, std, self.num_samples)  # (B,T,A)
        else:
            mean = best_actions.mean(0)  # (T,A=Nl+2)
            std = best_actions.std(0) + 0.05  # (T,A), +0.05 is added as a minimum std deviation (note pick threshold is 0.2)
            actions = np.zeros((self.num_samples, self.time_horizon, self.action_type+2))  # (B,T,A)
            for t in range(self.time_horizon):
                latent_means = mean[t, :-2]  # (Nl)
                latent_probs = (latent_means + 0.2) / (latent_means + 0.2).sum()  # (Nl)
                latent_idxs = np.random.choice(self.action_type, size=self.num_samples, p=latent_probs)  # (Nl)
                latent_one_hot = self.to_one_hot(latent_idxs, self.action_type)  # (B,Nl)
                actions[:, t, :-2] = latent_one_hot
            pick_locs = self.env.sample_multiple_place_locs_gaussian(mean[:, -2:], std[:, -2:], self.num_samples)  # (B,T,2)
            actions[:, :, -2:] = pick_locs  # (B,T,A)
        return actions

    # Input: vals (B), num_choices=Nc
    # Output: (B,Nc)
    def to_one_hot(self, vals, num_choices):
        return np.eye(num_choices)[vals]

    # Input: actions (B,A),  pred_recons (B,3,D,D)
    def plot_action_errors(self, env, actions, pred_recons, file_name):
        errors = env.get_action_error(ptu.get_numpy(actions))  # (B) np

        full_plot = pred_recons.view([5, -1] + list(pred_recons.shape[1:]))  # (5,B//5,3,D,D)
        caption = np.reshape(errors, (5, -1))  # (5,B//5) np
        plot_multi_image(ptu.get_numpy(full_plot), '{}/{}.png'.format(self.logging_dir, file_name), caption=caption)
#############End Action Selection Class#########


#############Start latent to pick location#########
# Input: env,  act_vals (B,K),  actions (B,A)
# Output: Image / heatmap of pick locations and corresponding act_vals
def plot_action_vals(env, act_vals, actions, file_name, is_normalized=False):
    file_name = file_name.split(".")[0]  # Removes .png or similar extention
    k = act_vals.shape[1]
    block_locs = env.get_block_locs()  # (Nb,3), where Nb=Number of blocks

    if is_normalized:
        normalized_act_vals = act_vals
    else:
        normalized_act_vals = act_vals/np.max(act_vals)  # (B,K)
    some_colors = cm.rainbow(np.linspace(0, 1, k))  # (K,4), rgba values

    # Regular 2D plot with boxes
    fig, ax = plt.subplots(1, figsize=(5 * 1.2, 3 * 1.2))
    # ax.imshow(cur_top_down_img, extent=[-2.5, 2.5, 1, 4], alpha=0.5)
    for i in range(k):
        rgba_colors = np.zeros((act_vals.shape[0], 4))  # (B,4)
        rgba_colors[:] = some_colors[i]  # (4)
        rgba_colors[:, -1] = np.minimum(normalized_act_vals[:, i]*2, 1)  # (B)
        ax.scatter(actions[:, 0], actions[:, 1], c=rgba_colors, label="{}".format(i))

        tmp = normalized_act_vals[:, i]
        mean_point = np.array([np.sum(tmp * actions[:, 0]),
                               np.sum(tmp * actions[:, 1])])
        mean_point = mean_point / np.sum(tmp)
        ax.scatter(mean_point[0], mean_point[1], color='k')
        ax.scatter(mean_point[0], mean_point[1], color=some_colors[i], marker="x")

    axes = plt.gca()
    axes.set_xlim([-2.5, 2.5])
    axes.set_ylim([1, 4])
    make_error_boxes(ax, block_locs[:, 0], block_locs[:, 1], env.TOLERANCE)
    plt.savefig("{}_2d.png".format(file_name))

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_batches = 100
    num_points_per_batch = actions.shape[0] // num_batches
    groups = defaultdict(list)
    for batch_ind in range(num_batches):
        start_ind = batch_ind * num_points_per_batch
        end_ind = (batch_ind + 1) * num_points_per_batch

        start_rand_k = np.random.randint(k)
        for i in range(k):
            tmp_act_vals = act_vals[start_ind:end_ind]
            tmp_actions = actions[start_ind:end_ind]

            i = (start_rand_k + i) % k
            rgba_colors = np.zeros((tmp_act_vals.shape[0], 4))  # (B,4)
            rgba_colors[:] = some_colors[i]  # (4)
            rgba_colors[:, -1] = 1  # tmp_act_vals[:, i]  # (B)
            tmp_label = ax.scatter(tmp_actions[:, 0], tmp_actions[:, 1], tmp_act_vals[:, i], c=rgba_colors, s=4)
            groups[i].append(tmp_label)

    tmp = [tuple(groups[i]) for i in range(k)]
    ax.legend(tmp, [i for i in range(k)], loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.savefig("{}_{}_3d.png".format(file_name, i))

# https://matplotlib.org/3.1.1/gallery/statistics/errorbars_and_boxes.html#sphx-glr-gallery-statistics-errorbars-and-boxes-py
def make_error_boxes(ax, xdata, ydata, box_width, linewidth=1, facecolor='none', edgecolor='g', alpha=1):
    errorboxes = []  # Create list for all the error patches

    # Loop over data points; create box from x,y points
    for x, y in zip(xdata, ydata):
        rect = Rectangle((x - box_width, y - box_width), box_width*2, box_width*2, linewidth=linewidth)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
    ax.add_collection(pc)  # Add collection to axes
    # art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")
#############End latent to pick location#########


########Start process env functions########
# Input: env_obs (D,D,3) or (T,D,D,3), values between 0-255, numpy array
# Output: (1,3,D,D), values between 0-1, torch
def process_env_obs(env_obs):
    if len(env_obs.shape) == 3: #(D,D,3) numpy
        env_obs = np.expand_dims(env_obs, 0) #(T=1,D,D,3), numpy
    return ptu.from_numpy(np.moveaxis(env_obs, 3, 1))/255


# Input: env_obs (A) or (T,A), numpy array
def process_env_actions(env_actions):
    if len(env_actions.shape) == 1: #(A) numpy
        env_actions = np.expand_dims(env_actions, 0) #(T=1,A), numpy
    return ptu.from_numpy(env_actions)
########End process env functions########


##############Start MPC Class ############
class Stage3_MPC:
    def __init__(self, model, logging_dir, accuracy_threshold):
        self.model = model
        self.logging_dir = logging_dir
        self.accuracy_threshold = accuracy_threshold

        if logging_dir is not None:
            if not os.path.exists(logging_dir):
                os.mkdir(logging_dir)

    # Inputs: goal_image (3,D,D), env, initial_obs (T1,3,D,D), initial_actions (T1-1,A), action_selection_class, T
    # Assume all images are between 0 & 1 and are tensors
    def run_plan(self, goal_image, env, initial_obs, initial_actions, action_selection_class,
                 num_actions_to_take, planning_horizon, true_data, filter_goal_image=None):
        #Goal inference
        self.env = env
        goal_info = self.goal_inference(goal_image, filter_goal_image)

        #State acquisition
        cur_state_and_other_info = self.state_acquisition(initial_obs, initial_actions)
        initial_recon = cur_state_and_other_info["final_recon"]

        #Planning
        actions_taken, actions_planned, pred_recons, obs, try_obs = [], [], [], [], []  #(T), (?,3,D,D), (T,D,D,3) np, (T,D,D,3) np, (T,D,D,3)
        pred_recons = [initial_recon[0]]
        best_accuracy = 0
        first_finished_plan_steps = np.nan

        for t in range(num_actions_to_take):
            next_actions, goal_latent_index, pred_recon = action_selection_class.select_action(goal_info, cur_state_and_other_info["state"],
                                                                                               env, self.model, "{}".format(t))  # (Tp,A), (Sc), (3,D,D)

            # try_obs.append(env.try_step(next_actions))  # (D,D,3)
            # pred_recons.append(pred_recon)  # (3,D,D)

            next_obs = [env.get_observation()]  # This is needed for update_state
            for i in range(planning_horizon):
                next_obs.append(env.step(next_actions[i]))  # (D,D,3), np
                try_obs.append(env.try_step(next_actions))  # (D,D,3)
                pred_recons.append(pred_recon)  # (3,D,D)

            actions_taken.extend(next_actions[:planning_horizon])  # (Tt,A)
            actions_planned.append(next_actions)  # (Tp,A)
            obs.extend(next_obs[1:])  # Don't want to include starting image again

            # next_obs = np.array([env.step(an_action) for an_action in next_actions]) #(Tp=1,D,D,3) 255's numpy
            # actions.extend(next_actions)
            # obs.extend(next_obs)

            # self._remove_goal_latent(goal_info, goal_latent_index)
            next_obs = np.array(next_obs)  # (Tp+1,D,D,3) np
            cur_state_and_other_info = self.update_state(next_obs, next_actions[:planning_horizon],
                                cur_state_and_other_info["state"], file_name="{}/state_update_{}.png".format(self.logging_dir, t))

            accuracy = self.env.compute_accuracy(true_data, threshold=self.accuracy_threshold)
            best_accuracy = max(accuracy, best_accuracy)
            if first_finished_plan_steps is np.nan and accuracy == 1:
                first_finished_plan_steps = t+1
                break


        ########Create final mpc image########
        # pdb.set_trace()
        goal_image_tensor = process_env_obs(goal_image)  # (1,3,D,D)
        starting_image_tensor = process_env_obs(initial_obs[-1])  # (1,3,D,D)

        obs = np.concatenate((initial_obs[-1:], obs))  # (T+1,D,D,3) np
        obs = process_env_obs(np.array(obs))  # (T+1,3,D,D)
        obs = torch.cat([obs, goal_image_tensor])  # (T+2,3,D,D)
        try_obs = process_env_obs(np.array(try_obs))  # (T,3,D,D)
        try_obs = torch.cat([starting_image_tensor, try_obs, goal_image_tensor])  # (T+2,3,D,D)
        pred_recons = torch.stack(pred_recons)  # (T+1,3,D,D)
        pred_recons = torch.cat([pred_recons, goal_info["final_recon"]])  # (T+2,3,D,D)
        save_image(torch.cat([obs, pred_recons, try_obs], dim=0), "{}/mpc.png".format(self.logging_dir), nrow=obs.shape[0])


        ########Compute result stats########
        final_obs = process_env_obs(env.get_observation())  # (1,3,D,D)
        torch_goal_image = process_env_obs(goal_image)  # (1,3,D,D)
        mse = ptu.get_numpy(torch.pow(final_obs - torch_goal_image, 2).mean())  # Compare final obs to goal obs (Sc), numpy

        #correct = env.compute_accuracy(true_data, threshold=self.accuracy_threshold)
        stats = {'mse': mse, 'correct': best_accuracy, 'actions': actions_taken, "first_finished_plan_steps": first_finished_plan_steps}
        return stats


    def _filter_goal_image(self, goal_info, n_objects):
        goal_latents_mask = goal_info["masks"].squeeze(0).squeeze(1)  # (B=1,K,1,D,D) -> (K,1,D,D) -> (K,D,D)
        goal_latents_recon = goal_info["sub_images"].squeeze(0)  # (B=1,K,3,D,D) -> (K,3,D,D)

        vals, keep = torch.sort(goal_latents_mask.mean((1, 2)), descending=True)
        save_image(goal_latents_mask[keep].unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_goal_masks.png'.format(self.logging_dir))

        blank_image = process_env_obs(self.env._blank_observation)  # (1,3,D,D)
        blank_image = blank_image * goal_latents_mask.unsqueeze(1)  # (1,3,D,D)*(K,1,D,D) -> (K,3,D,D)

        save_image(blank_image, '{}/filter_blank_image.png'.format(self.logging_dir))
        save_image(goal_latents_recon, '{}/filter_blank_image_2.png'.format(self.logging_dir))

        difs = torch.abs(goal_latents_recon - blank_image)  # (K,3,D,D)
        difs = torch.where(difs > 10 / 255, difs, ptu.zeros_like(difs))  # .sum((1, 2)) #(K,3,D,D)
        difs = difs.sum(1)  # (K,D,D)
        save_image(difs.unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_difs.png'.format(self.logging_dir))
        difs = difs.sum((1, 2))  # (K)

        vals, keep = torch.sort(difs, descending=True)

        save_image(goal_latents_mask[keep].unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_goal_masks_sorted.png'.format(self.logging_dir))

        keep = keep[:n_objects]
        goal_info["state"]["post"]["samples"] = goal_info["state"]["post"]["samples"][:, keep]  # (B=1,K,R) -> (B=1,N_ob,R)
        goal_info["sub_images"] = goal_info["sub_images"][:, keep]  # (B=1,N_ob,3,D,D)
        goal_info["masks"] = goal_info["masks"][:, keep]  # (B,N_ob,1,D,D)
        goal_info["colors"] = goal_info["colors"][:, keep] # (B=1,N_ob,3,D,D)

        save_image(goal_info["sub_images"].squeeze(0), '{}/filter_mpc_goal_latents_recon.png'.format(self.logging_dir), nrow=10)
        return goal_info

    def _remove_goal_latent(self, goal_info, latent_index):
        def exclude_one(torch_array, index):
            if torch_array.shape[1] == 1:
                return None
            return torch.stack([torch_array[:, i] for i in range(torch_array.shape[1]) if i != index], dim=1)

        # pdb.set_trace()
        goal_info["colors"] = exclude_one(goal_info["colors"], latent_index) # (B=1,K,3,D,D)
        goal_info["masks"] = exclude_one(goal_info["masks"], latent_index)  # (B=1,K,1,D,D)
        goal_info["sub_images"] = exclude_one(goal_info["sub_images"], latent_index)  # (B=1,K,3,D,D)
        goal_info["state"]["post"]["samples"] = exclude_one(goal_info["state"]["post"]["samples"], latent_index)
        return goal_info


    #Input: numpy array goal_image (D,D,3)
    def goal_inference(self, goal_image, filter_goal_image):
        schedule = np.zeros(5) #5 refinement steps on goal image
        input_goal_image = process_env_obs(goal_image).unsqueeze(0) #(B=1,T=1,3,D,D)
        num_tries = 40
        input_goal_image = input_goal_image.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        goal_info = self.model.batch_internal_inference(input_goal_image, None, None, schedule, select_best=True,
                                                        figure_path=self.logging_dir + "/goal_inference_general.png")

        goal_info["goal_image"] = process_env_obs(goal_image) #(B=1,3,D,D)

        if filter_goal_image:
            goal_info = self._filter_goal_image(goal_info, **filter_goal_image)
        else:
            visualizer.visualize_state_info(goal_info, file_name="{}/goal_inference_selected.png".format(self.logging_dir), true_image=input_goal_image[0])
        return goal_info

    #Inputs: obs (T1,D,D,3) np, actions (T1-1,A) or None np
    def state_acquisition(self, obs, actions):
        input_obs = process_env_obs(obs).unsqueeze(0) #(B=1,T1,3,D,D)
        num_tries = 40
        input_obs = input_obs.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        if actions is not None:
            input_actions = process_env_actions(actions).unsqueeze(0) #(B=1,T1-1,A)
            input_actions = input_actions.repeat(num_tries, 1, 1)  # (Nt,T1,A)
        else:
            input_actions = None
        schedule = self.model.get_rprp_schedule(seed_steps=4, num_images=obs.shape[0], num_refine_per_physics=2)
        state_info = self.model.batch_internal_inference(input_obs, input_actions, None, schedule,
                                                         select_best=True, figure_path="{}/state_acquisition.png".format(self.logging_dir))
        # visualizer.visualize_state_info(state_info, file_name="{}/state_acquisition.png".format(self.logging_dir), true_image=input_obs[:, -1])
        return state_info


    #Input: next_obs (T1,D,D,3) np, next_actions (T1,A) np, cur_state
    #  Note: next_obs[0] is the initial image that cur_state should decode to (so we don't need to do initial seed steps on that)
    def update_state(self, next_obs, next_actions, cur_state, file_name=None):
        input_obs = process_env_obs(next_obs).unsqueeze(0)  # (B=1,T1,3,D,D)
        num_tries = 40
        input_obs = input_obs.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        input_actions = process_env_actions(next_actions).unsqueeze(0)  # (B=1,T1,A)
        input_actions = input_actions.repeat(num_tries, 1, 1)  # (Nt,T1,A)

        schedule = self.model.get_rprp_schedule(seed_steps=0, num_images=next_obs.shape[0], num_refine_per_physics=3)
        state_info = self.model.batch_internal_inference(input_obs, input_actions, cur_state, schedule,
                                                         select_best=True, figure_path=file_name)
        # if file_name is not None:
        #     visualizer.visualize_state_info(state_info, file_name, true_image=input_obs[:, -1])
        return state_info
##############End MPC Class ############



#########Loading data and running mpc#########
def copy_to_save_file(dir_str):  #RV: TODO: FIX THIS
    base = get_module_path()
    shutil.copytree(base + '/rlkit/torch/iodine', dir_str+'/saved_torch_iodine_files')
    shutil.copy2(base + '/examples/mpc_v2/stage3/stage3_mpc.py', dir_str + '/saved_stage3_mpc.py')

def main(variant):
    from op3.core import logger
    #copy_to_save_file(logger.get_snapshot_dir())
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    module_path = get_module_path()

    ######Start Model loading######
    op3_args = variant["op3_args"]
    op3_args['K'] = 4
    m = op3_model.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=4)

    model_file = module_path + '/exps/pickplace_exps/saved_models/{}.pkl'.format(variant['model_file'])
    state_dict = torch.load(model_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)
    m.cuda()
    m.eval_mode = True
    ######End Model loading######

    ######Start goal info loading######
    goal_idxs = range(variant["goal_start_end_range"][0], variant["goal_start_end_range"][1])
    stats = {'mse': [], 'correct': [], 'actions': [], 'first_finished_plan_steps': []}

    goal_folder = module_path + '/data/goals/pickplace_goals/objects_seed_{}/'.format(variant['number_goal_objects'])
    num_seed_frames = 3
    aggregate_stats = {}
    goals_tried = 0
    ######End goal info loading######

    ######Start planning execution######
    for goal_idx in goal_idxs:
        env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64,
                                   include_z=False)  # Note num_objects & num_colors do not matter

        ####Load goal and starting info
        with open(goal_folder + 'goal_data.pkl', 'rb') as f:
            goal_dict = pickle.load(f)
        goal_image = goal_dict["goal_image"][goal_idx]  # (D,D,3) np
        # frames = goal_dict["frames"][i]  # (T,D,D,3) np
        actions = goal_dict["actions"][goal_idx]  # (T-1,6) np
        seed_actions = env._post_process_actions(actions)  # (T-1,4) np
        goal_env_info = goal_dict["goal_env_info"][goal_idx]
        starting_env_info = goal_dict["starting_env_info"][goal_idx]

        #####Get seed steps
        env.set_env_info(starting_env_info)
        seed_frames = [env.get_observation()]
        if num_seed_frames > 1:
            seed_actions = seed_actions[:num_seed_frames-1]
            for an_action in seed_actions:
                seed_frames.append(env.step(an_action))
        else:
            seed_actions = None
        seed_frames = np.array(seed_frames)  # (T,D,D,3) np
        if env.get_block_locs(check_all_in_bounds=True) is False:
            continue
        goals_tried += 1


        #####Set up mpc
        logging_directory = "{}/goal_{}".format(logger.get_snapshot_dir(), goal_idx)
        cost_class = Cost(logging_directory, **variant['cost_args'])
        cem_process = Stage3_CEM(logging_dir=logging_directory, score_actions_class=cost_class, **variant['cem_args'])

        mpc = Stage3_MPC(m, logging_directory, variant['accuracy_threshold'])
        single_stats = mpc.run_plan(goal_image, env, seed_frames, seed_actions, cem_process, num_actions_to_take=variant["num_actions_to_take"],
                                    planning_horizon=variant["num_action_to_take_per_plan"], true_data=goal_env_info, filter_goal_image = False)
                                    # filter_goal_image={"n_objects": variant['number_goal_objects']})
        for k, v in single_stats.items():
            stats[k].append(v)

        with open(logger.get_snapshot_dir() + '/results_rolling.pkl', 'wb') as f:
            pickle.dump(stats, f)

        for k, v in stats.items():
            if k != 'actions':
                aggregate_stats[k] = float(np.nanmean(v))
        aggregate_stats["individual_correct"] = stats["correct"]
        aggregate_stats["num_goals_tried"] = goals_tried
        aggregate_stats["individual_first_finished_plan_steps"] = stats["first_finished_plan_steps"]
        json.dump(aggregate_stats, open(logger.get_snapshot_dir() + '/results_stats.json', 'w'))

    with open(logger.get_snapshot_dir() + '/results_final.pkl', 'wb') as f:
        pickle.dump(stats, f)
##############End planning execution#############


# Example use case: CUDA_VISIBLE_DEVICES=2 python mpc_pickplace.py -de 0 -m [MODEL_NAME]  #curriculum_aws_params
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-de', '--debug', type=int, default=0)
    parser.add_argument('-m', '--model_file', type=str, required=True)
    parser.add_argument('-mode', '--mode', type=str, default='here_no_doodad')
    parser.add_argument('-n', '--num_obs', type=str, default='here_no_doodad')
    args = parser.parse_args()

    num_goals = 100
    num_goal_objects = 1

    variant = dict(
        algorithm='MPC',
        op3_args=params_to_info[args.model_file]["op3_args"],
        accuracy_threshold=0.3,
        cem_args=dict(
            cem_steps=5,
            num_samples=200,
            time_horizon=num_goal_objects,
            action_type= 4,  # Should be [None, 4], This controls if it is object oriented (when set to 4) or raw env action space (None)
        ),
        num_actions_to_take=num_goal_objects*2,
        num_action_to_take_per_plan=num_goal_objects,
        cost_args=dict(
            core_type='final_recon',  # "subimage", "final_recon", "latent"
            compare_func='mse',    # mse, psuedo_intersect
            post_process='raw',
            aggregate='sum',
        ),
        goal_start_end_range=[0, 100],
        debug=args.debug,
        model_file=args.model_file,
        number_goal_objects=num_goal_objects,
    )

    run_experiment(
        main,
        exp_prefix='iodine-mpc-stage3-n{}-{}-v2'.format(variant["number_goal_objects"], args.model_file),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        region='us-west-2',
    )






