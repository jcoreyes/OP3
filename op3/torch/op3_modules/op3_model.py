import torch
import torch.utils.data
from op3.torch.op3_modules.physics_network import PhysicsNetwork, PhysicsNetwork_v2_No_Sharing, Physics_Args
from op3.torch.op3_modules.refinement_network import RefinementNetwork, Refinement_Args, RefinementNetwork_v2_No_Sharing
from op3.torch.op3_modules.decoder_network import DecoderNetwork, Decoder_Args, DecoderNetwork_v2_No_Sharing
from op3.torch.op3_modules.visualizer import quicksave, visualize_state_info
from torch.distributions.multivariate_normal import MultivariateNormal

from torch import nn
from torch.nn import functional as F, Parameter
from op3.pythonplusplus import identity
from op3.torch import pytorch_util as ptu
import numpy as np
from op3.torch.modules import LayerNorm2D, LayerNorm
from op3.core import logger
import os
import pdb


# Variant must contain the following keywords: refinement_model_type, decoder_model_type, dynamics_model_type, K
#   Note: repsize represents the size of the deterministic & stochastic each, so the full state size is repsize*2
def create_model_v2(variant, det_size, sto_size, action_dim):
    K = variant['K']

    ref_model_args = Refinement_Args[variant["refinement_model_type"]]
    if ref_model_args[0] == "reg":
        refinement_kwargs = ref_model_args[1](sto_size)
        refinement_net = RefinementNetwork(**refinement_kwargs)
    elif ref_model_args[0] == "sequence_iodine":
        refinement_kwargs = ref_model_args[1](sto_size, action_dim)
        refinement_net = RefinementNetwork(**refinement_kwargs)
    elif ref_model_args[0] == "no_share":
        refinement_kwargs = ref_model_args[1](sto_size, K)
        refinement_net = RefinementNetwork_v2_No_Sharing(**refinement_kwargs)
    else:
        raise ValueError("{}".format(ref_model_args[0]))

    dec_model_args = Decoder_Args[variant["decoder_model_type"]]
    if dec_model_args[0] == "reg":
        decoder_kwargs = dec_model_args[1](det_size + sto_size)
        decoder_net = DecoderNetwork(**decoder_kwargs)
    elif dec_model_args[0] == "reg_no_share":
        decoder_kwargs = dec_model_args[1](det_size + sto_size, K)
        decoder_net = DecoderNetwork_v2_No_Sharing(**decoder_kwargs)
    else:
        raise ValueError("{}".format(dec_model_args[0]))

    dyn_model_args = Physics_Args[variant["dynamics_model_type"]]
    if dyn_model_args[0] == "reg":
        physics_kwargs = dyn_model_args[1](det_size, sto_size, action_dim)
        dynamics_net = PhysicsNetwork(**physics_kwargs)
    elif dyn_model_args[0] == "reg_no_share":
        physics_kwargs = dyn_model_args[1](det_size, sto_size, action_dim, K)
        dynamics_net = PhysicsNetwork_v2_No_Sharing(**physics_kwargs)
    else:
        raise ValueError("{}".format(variant["dynamics_model_type"][0]))

    model = OP3Model(refinement_net, dynamics_net, decoder_net, det_size, sto_size, **variant['extra_args'])
    model.set_k(K)
    return model


# Notation: R denotes size of deterministic & stochastic state (each), R2 denotes dize of lstm hidden state for iodine
#  D denotes image size, A denotes action dimension, B denotes batch size
#  (Sc) denotes a scalar while (X,Y,Z) represent the shape
class OP3Model(torch.nn.Module):
    def __init__(self, refine_net, dynamics_net, decode_net, det_size, sto_size, beta=1, deterministic_sampling=False):
        super().__init__()
        self.refinement_net = refine_net
        self.dynamics_net = dynamics_net
        self.decode_net = decode_net
        self.K = None
        self.det_size = det_size
        self.sto_size = sto_size
        self.full_rep_size = det_size + sto_size
        self.deterministic_sampling = deterministic_sampling  # Use the mean instead of actually sampling from the distribution

        # Loss hyper-parameters
        self.sigma = 0.1  # Sigma for reconstruction loss
        self.set_beta(beta)  # Beta is the coefficient on the KL loss

        # Refinement variables
        l_norm_sizes_2d = [1, 1, 1, 3]
        self.layer_norms_2d = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes_2d])
        l_norm_sizes_1d = [self.sto_size, self.sto_size]
        self.layer_norms_1d = nn.ModuleList([LayerNorm(l, center=True, scale=True) for l in l_norm_sizes_1d])

        self.eval_mode = False # Should set to true when testing

        # Initial state parameters
        if det_size != 0:
            self.inital_deter_state = Parameter(ptu.randn((det_size))/np.sqrt(det_size))
        self.initial_lambdas1 = Parameter(ptu.randn((sto_size))/np.sqrt(sto_size))
        self.initial_lambdas2 = Parameter(ptu.randn((sto_size))/np.sqrt(sto_size))

        self.debug = {}

    def set_k(self, k):
        self.K = k
        self.dynamics_net.set_k(k)

    def set_beta(self, beta):
        if self.deterministic_sampling and beta != 0:
            print("DANGER ALERT! Having beta set to {} is mathematically incorrect when sampling deterministically".format(beta))
        self.beta = beta

    #Input: x: (a,b,*)
    #Output: y: (a*b,*)
    def _flatten_first_two(self, x):
        if x is None:
            return x
        return x.view([x.shape[0]*x.shape[1]] + list(x.shape[2:]))

    #Input: x: (bs*k,*)
    #Output: y: (bs,k,*)
    def _unflatten_first(self, x, k):
        if x is None:
            return x
        return x.view([-1, k] + list(x.shape[1:]))

    #Input: inputs (B*K,3,D,D), targets (B*K,3,D,D), sigma (Sc)
    #Output: (B*K,D,D)
    def _gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (np.sqrt(sigma ** (2 * ch)) * 248.05)

    #Input: softplus (*)
    #Output: stds (*)
    def _softplus_to_std(self, softplus):
        softplus = torch.min(softplus, torch.ones_like(softplus)*80)
        return torch.sqrt(torch.log(1 + softplus.exp()) + 1e-5)


    #Compute the kl loss between prior and posterior
    #Note: This is NOT normalized
    def kl_divergence_prior_post(self, prior, post):
        mu1, softplus1 = prior["lambdas1"], prior["lambdas2"]
        mu2, softplus2 = post["lambdas1"], post["lambdas2"]

        stds1 = self._softplus_to_std(softplus1)
        stds2 = self._softplus_to_std(softplus2)
        q1 = MultivariateNormal(loc=mu1, scale_tril=torch.diag_embed(stds1))
        q2 = MultivariateNormal(loc=mu2, scale_tril=torch.diag_embed(stds2))
        return torch.distributions.kl.kl_divergence(q2, q1)  # KL(post||prior), note ordering matters

    def get_samples(self, means, softplusses):
        if self.deterministic_sampling:
            return means
        stds = self._softplus_to_std(softplusses)
        epsilon = ptu.randn(*means.size()).to(stds.device)
        latents = epsilon * stds + means
        return latents

    def get_full_tensor_state(self, hidden_state):
        if self.det_size == 0:
            return hidden_state["post"]["samples"]
        return torch.cat([hidden_state["post"]["deter_state"], hidden_state["post"]["samples"]], dim=2) #(B,K,R)

    #Input: Integer n denoting how many hidden states to initialize
    #Output: Returns hidden states, tuples of (n*k,repsize) and (n*k,lstm_size)
    def _get_initial_hidden_states(self, n, device):
        k = self.K

        if self.det_size == 0:
            deter_state = None
        else:
            deter_state = self._unflatten_first(self.inital_deter_state.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rd)

        lambdas1 = self._unflatten_first(self.initial_lambdas1.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rs)
        lambdas2 = self._unflatten_first(self.initial_lambdas2.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rs)
        samples = self.get_samples(lambdas1, lambdas2) #(n,k,Rd+Rs)
        h1, h2 = self.refinement_net.initialize_hidden(n * k)  # Each (1,n*k,lstm_size)
        h1, h2 = h1.to(device), h2.to(device)

        hidden_state = {
            "prior" : {
                "lambdas1": torch.zeros((n, k, self.sto_size)).to(device),  # (n*k,Rs)
                "lambdas2": torch.log(torch.exp(torch.ones((n, k, self.sto_size))) - 1).to(device), #log(e-1) as softplus (n*k,Rs)
            },
            "post" : {
                "deter_state": deter_state, #(n*k,Rd)
                "lambdas1" : lambdas1, #self._unflatten_first(lambdas1, k),   #(n*k,Rs)
                "lambdas2" : lambdas2, #self._unflatten_first(lambdas2, k),   #(n*k,Rs)
                "extra_info" : [h1, h2], # Each (1,n*k,lstm_size)
                "samples" : samples      #(n,k,Rd+Rs)
            }
        }
        return hidden_state


    #Inputs: hidden_states, images (B,3,D,D), action (B,A) or None, previous_decode_loss_info
    #Outputs: new_hidden_states
    #  Updates posterior lambda but not prior or posterior deter_state
    def refine(self, hidden_states, images, action=None, previous_decode_loss_info=None):
        bs, imsize = images.shape[0], images.shape[2]
        K = self.K
        tiled_k_shape = (bs*K, -1, imsize, imsize)

        if previous_decode_loss_info is None:
            colors, mask, mask_logits = self.decode(hidden_states)  #colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
            color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss = \
                self.get_loss(hidden_states, colors, mask, images)
            #color_probs(B, K, D, D), pixel_complete_log_likelihood(B, D, D), kle_loss(Sc), complete_log_likelihood (Sc), total_loss (Sc)
        else:
            colors, mask, mask_logits = previous_decode_loss_info[0]
            color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss = previous_decode_loss_info[1]

        posterior_mask = color_probs / (color_probs.sum(1, keepdim=True) + 1e-8)  #(B,K,D,D)
        leave_out_ll = pixel_complete_log_likelihood.unsqueeze(1) - mask.squeeze(2) * color_probs #(B,K,D,D)

        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
            torch.autograd.grad(total_loss, [colors, mask, hidden_states["post"]["lambdas1"], hidden_states["post"]["lambdas2"]], create_graph=not self.eval_mode,
                                retain_graph=not self.eval_mode)

        k_images = images.unsqueeze(1).repeat(1, K, 1, 1, 1) #(B,K,3,D,D)

        lns_2d = self.layer_norms_2d
        a = (torch.cat([
                k_images.view(tiled_k_shape), # (B*K,3,D,D)
                colors.view(tiled_k_shape), # (B*K,3,D,D)
                mask.view(tiled_k_shape),  # (B*K,1,D,D)
                mask_logits.view(tiled_k_shape), # (B*K,1,D,D)
                posterior_mask.view(tiled_k_shape), #(B*K,1,D,D)
                lns_2d[0](mask_grad.view(tiled_k_shape).detach()),  # (B*K,1,D,D)
                lns_2d[1](pixel_complete_log_likelihood.unsqueeze(1).repeat(1, K, 1, 1).view(tiled_k_shape).detach()), # (B*K,1,D,D)
                lns_2d[2](leave_out_ll.view(tiled_k_shape).detach()), # (B*K,1,D,D)
                lns_2d[3](x_hat_grad.view(tiled_k_shape).detach())], # (B*K,3,D,D)
            1))  # (B*K,3+3+1+1+1+1+1+1+3,D,D) -> (B*K,15,D,D)


        lns_1d = self.layer_norms_1d
        extra_input = torch.cat([lns_1d[0](lambdas_grad_1.view(bs * K, -1).detach()), #(B*K,Rs)
                                 lns_1d[1](lambdas_grad_2.view(bs * K, -1).detach()) #(B*K,Rs)
                                 ], -1) #(B*K,2*Rs)

        if action is not None: #Use action as extra input into refinement: This is only for next step refinement (sequence iodine)
            action = self._flatten_first_two(action.unsqueeze(1).repeat(1, K, 1)) #(B,A)->(B,K,A)->(B*K,A)

        h1, h2 = hidden_states["post"]["extra_info"][0], hidden_states["post"]["extra_info"][1] #Each (1,B*K,R2)
        h1 = h1.view(bs * K, -1)
        h2 = h2.view(bs * K, -1)

        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat(
                                                             [extra_input, self._flatten_first_two(hidden_states["post"]["lambdas1"]),
                                                              self._flatten_first_two(hidden_states["post"]["lambdas2"]),
                                                              self._flatten_first_two(hidden_states["post"]["samples"])], -1),
                                                         add_fc_input=action) #Lambdas (B*K,Rs),   h (B*K,R2)

        lambdas1 = self._unflatten_first(lambdas1, K) #(B,K,Rs)
        lambdas2 = self._unflatten_first(lambdas2, K) #(B,K,Rs)
        samples = self.get_samples(lambdas1, lambdas2) #(B,K,Rs)
        new_hidden_states = {
            "prior": hidden_states["prior"],  # Do not change prior
            "post": {  # Update post
                "deter_state": hidden_states["post"]["deter_state"], #Do not update deterministic part of state (B,K,R)
                "lambdas1": lambdas1, #Update lambdas (B,K,R)
                "lambdas2": lambdas2, #(B,K,R)
                "extra_info": [h1, h2], #Update refinement lstm args, each (1,B*K,R2)
                "samples": samples #Update samples (B,K,R)
            }
        }
        return new_hidden_states

    # Inputs: hidden_states, actions (B,A) or None
    # Outputs: new_hidden_states
    #   Update prior and posterior distribution
    def dynamics(self, hidden_states, actions):
        b, K = hidden_states["post"]["samples"].shape[:2]
        if actions is not None:
            actions = actions.unsqueeze(1).repeat(1, K, 1)  # (B,K,A)
            actions = self._flatten_first_two(actions)  # (B*K,A)
        full_states = self._flatten_first_two(self.get_full_tensor_state(hidden_states)) #(B*K,Rs+Rd)

        deter_states, lambdas1, lambdas2 = self.dynamics_net(full_states, actions) #(B*K,Rd), (B*K,Rs), (B*K,Rs)
        h1, h2 = ptu.zeros_like(hidden_states["post"]["extra_info"][0]).to(hidden_states["post"]["extra_info"][0].device), \
                 ptu.zeros_like(hidden_states["post"]["extra_info"][1]).to(hidden_states["post"]["extra_info"][1].device) #Set the h's to zero as the next refinement should start from scratch (B,K,R2)

        lambdas1 = self._unflatten_first(lambdas1, K)  # (B,K,Rs)
        lambdas2 = self._unflatten_first(lambdas2, K)  # (B,K,Rs)
        samples = self.get_samples(lambdas1, lambdas2)  # (B,K,Rs)
        new_hidden_states = {
            "prior": { #Update prior
                "lambdas1": lambdas1,  # (B,K,Rs) ##NOTE: TRY Detaching or not
                "lambdas2": lambdas2,  # (B,K,Rs)
            },
            "post": { #Update posterior
                "deter_state": self._unflatten_first(deter_states, K), #(B,K,Rd)
                "lambdas1": lambdas1,  # (B,K,Rs)
                "lambdas2": lambdas2,  # (B,K,Rs)
                "extra_info": [h1, h2],  # Each (B,K,R2)
                "samples": samples # (B,K,Rs)
            }
        }
        return new_hidden_states

    # Input: hidden_states
    # Output: colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
    def decode(self, hidden_states):
        bs, k = hidden_states["post"]["samples"].shape[:2]
        full_states = self._flatten_first_two(self.get_full_tensor_state(hidden_states)) #(B*K,Rs+Rd)
        mask_logits, colors = self.decode_net(full_states) #mask_logits: (B*K,1,D,D),  colors: (B*K,3,D,D)

        mask_logits = self._unflatten_first(mask_logits, k) #(B,K,1,D,D)
        mask = F.softmax(mask_logits, dim=1)  #(B,K,1,D,D), these are the mask probability values
        colors = self._unflatten_first(colors, k) #(B,K,3,D,D)
        return colors, mask, mask_logits

    # Inputs: colors (B,K,3,D,D),  masks (B,K,1,D,D),  target_imgs (B,3,D,D)
    # Outputs: color_probs (B,K,D,D), pixel_complete_log_likelihood (B,D,D), kle_loss (Sc),
    #   complete_log_likelihood (Sc), total_loss (Sc)
    def get_loss(self, hidden_states, colors, mask, target_imgs):
        b, k = colors.shape[:2]
        k_targs = target_imgs.unsqueeze(1).repeat(1, k, 1, 1, 1)  # (B,3,D,D) -> (B,1,3,D,D) -> (B,K,3,D,D)
        k_targs = self._flatten_first_two(k_targs)  # (B,K,3,D,D) -> (B*K,3,D,D)
        tmp_colors = self._flatten_first_two(colors)  # (B,K,3,D,D) -> (B*K,3,D,D)
        color_probs = self._gaussian_prob(tmp_colors, k_targs, self.sigma)  # Computing p(x|h),  (B*K,D,D)
        color_probs = self._unflatten_first(color_probs, k)  # (B,K,D,D)
        pixel_complete_log_likelihood = (mask.squeeze(2) * color_probs).sum(1)  # Sum over K, pixelwise complete log likelihood (B,D,D)
        complete_log_likelihood = -torch.log(pixel_complete_log_likelihood + 1e-12).sum() / b  # (Scalar)

        kle = self.kl_divergence_prior_post(hidden_states["prior"], hidden_states["post"])
        kle_loss = kle.sum() / b  # KL loss, (Sc)

        total_loss = complete_log_likelihood + self.beta * kle_loss  # Total loss, (Sc)
        return color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss

    # Inputs: images: None or (B, T_obs, 3, D, D),  actions: None or (B, T_acs, A),  initial_hidden_state or None
    #   schedule: (T1),   loss_schedule:(T1)
    # Output: colors_list (B,T1,K,3,D,D), masks_list (B,T1,K,1,D,D), final_recon (B,3,D,D),
    #   total_loss, total_kle_loss, total_clog_prob, mse are all (Sc), end_hidden_state
    def run_schedule(self, images, actions, initial_hidden_state, schedule, loss_schedule, should_detach=False):
        self.debug["schedule"] = schedule
        self.debug["at"] = -1

        if images is not None:
            b = images.shape[0]
        elif actions is not None:
            b = actions.shape[0]
        else:
            raise ValueError("Need either images or actions")


        if initial_hidden_state is None: #Initialize initial_hidden_state if it is not passed in
            initial_hidden_state = self._get_initial_hidden_states(b, images.device)

        ###Initialize variables for saving output
        #Save outputs: colors_list (T1,B,K,3,D,D),  masks (T1,B,K,1,D,D),  losses_list (T1+1)
        colors_list, masks_list, losses_list, kle_loss_list, clog_prob_list = [], [], [], [], []
        current_step = 0
        cur_hidden_state = initial_hidden_state
        previous_decode_loss_info = None #Initial loss for initial lambda parameters

        ###Loss based on schedule
        for i in range(len(schedule)):
            self.debug["at"] = i
            if schedule[i] == 0:  # Refinement step
                input_img = images[:, current_step]  # (B,3,D,D)
                cur_hidden_state = self.refine(cur_hidden_state, input_img, previous_decode_loss_info=previous_decode_loss_info)
            elif schedule[i] == 1:  # Dynamics step
                if actions is not None:
                    input_actions = actions[:, current_step]  # (B,A)
                else:
                    input_actions = None
                cur_hidden_state = self.dynamics(cur_hidden_state, input_actions)
                current_step += 1
            elif schedule[i] == 2:  # Next step refinement, just used for sequence iodine
                if actions is not None:
                    input_actions = actions[:, current_step]  # (B,A)
                else:
                    input_actions = None
                input_img = images[:, current_step]
                cur_hidden_state = self.refine(cur_hidden_state, input_img, action=input_actions,
                                               previous_decode_loss_info=previous_decode_loss_info)
                current_step += 1
            else:
                raise ValueError("Invalid schedule entry: {}".format(schedule[i]))

            colors, mask, mask_logits = self.decode(cur_hidden_state)
            colors_list.append(colors)
            masks_list.append(mask)

            if loss_schedule[i] != 0:  # Calculate the loss if we need to do so
                target_images = images[:, current_step]  # (B,3,D,D)
                color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss = \
                    self.get_loss(cur_hidden_state, colors, mask, target_images)

                losses_list.append(total_loss * loss_schedule[i])
                kle_loss_list.append(kle_loss * loss_schedule[i])
                clog_prob_list.append(clog_prob * loss_schedule[i])

                previous_decode_loss_info = [[colors, mask, mask_logits],
                                      [color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss]]
            else:
                previous_decode_loss_info = None

        colors_list = torch.stack(colors_list) #(T1,B,K,3,D,D)
        masks_list = torch.stack(masks_list) #(T1,B,K,1,D,D)

        if sum(loss_schedule) == 0:
            total_loss, total_kle_loss, total_clog_prob = None, None, None
        else:
            sum_loss_weights = sum(loss_schedule)
            total_loss = sum(losses_list) / sum_loss_weights #Scalar
            total_kle_loss = sum(kle_loss_list) / sum_loss_weights
            total_clog_prob = sum(clog_prob_list) / sum_loss_weights


        final_recon = (colors_list[-1] * masks_list[-1]).sum(1) #(B,K,3,D,D) -> (B,3,D,D)
        if images is not None:
            mse = torch.pow(final_recon - images[:, -1], 2).mean() #(B,3,D,D) -> (Sc)
        else:
            mse = torch.zeros(size=()).to(final_recon.device)  # (Sc)
        colors_list = colors_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,3,D,D) -> (B,T1,K,3,D,D)
        masks_list = masks_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,1,D,D) -> (B,T1,K,1,D,D)


        #This part is needed for dataparallel as all tensors need to be (B,*)
        tmp = [cur_hidden_state["post"]["extra_info"][0].view(b, self.K, -1), cur_hidden_state["post"]["extra_info"][1].view(b, self.K, -1)]
        cur_hidden_state["post"]["extra_info"] = tmp


        if should_detach:
            colors_list = colors_list.detach()
            masks_list = masks_list.detach()
            final_recon = final_recon.detach()
            total_loss = total_loss.detach() if total_loss is not None else None
            total_kle_loss = total_kle_loss.detach() if total_kle_loss is not None else None
            total_clog_prob = total_clog_prob.detach() if total_clog_prob is not None else None
            mse = mse.detach()
            cur_hidden_state = self.detach_state(cur_hidden_state)

        return colors_list, masks_list, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state

    #Wrapper for self.run_schedule() required for training with DataParallel
    def forward(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        return self.run_schedule(images, actions, initial_hidden_state, schedule, loss_schedule)
    #######################End core functions for training########################



    #######Start extra functions not needed for training but useful for testing/mpc########
    #Inputs: seed_steps, num_images, num_refine_per_physics all (Sc)
    #Outputs: schedule (T) numpy array
    def get_rprp_schedule(self, seed_steps, num_images, num_refine_per_physics):
        schedule = np.zeros(seed_steps + (num_images-1) * (num_refine_per_physics+1))
        schedule[seed_steps::(num_refine_per_physics+1)] = 1
        return schedule

    #Input: Hidden state
    #Output: Hidden state with everything detached
    def detach_state(self, cur_hidden_state):
        tmp = [cur_hidden_state["post"]["extra_info"][0].detach(),
               cur_hidden_state["post"]["extra_info"][1].detach()]

        detached_hidden_state = {
            "prior": {
                "lambdas1": cur_hidden_state["prior"]["lambdas1"].detach(),  # (B,K,R)
                "lambdas2": cur_hidden_state["prior"]["lambdas2"].detach(),  # (B,K,R)
            },
            "post": {  # Update posterior
                "deter_state": cur_hidden_state["post"]["deter_state"].detach() if cur_hidden_state["post"]["deter_state"] is not None else None,
                # (B,K,R)
                "lambdas1": cur_hidden_state["post"]["lambdas1"].detach(),  # (B,K,R)
                "lambdas2": cur_hidden_state["post"]["lambdas2"].detach(),  # (B,K,R)
                "extra_info": tmp,  # Each (B*K,R2)
                "samples": cur_hidden_state["post"]["samples"].detach()  # (B,K,R)
            }
        }
        return detached_hidden_state

    #Inputs: cur_hidden_state with B=1, n
    #Outputs: cur_hidden_state with B=n
    def replicate_state(self, cur_hidden_state, n):
        if cur_hidden_state is None:
            return None
        tmp = [cur_hidden_state["post"]["extra_info"][0].repeat(n,1,1),
               cur_hidden_state["post"]["extra_info"][1].repeat(n,1,1)]

        new_hidden_state = {
            "prior": {
                "lambdas1": cur_hidden_state["prior"]["lambdas1"].repeat(n,1,1),  # (n,K,R)
                "lambdas2": cur_hidden_state["prior"]["lambdas1"].repeat(n,1,1),  # (n,K,R)
            },
            "post": {  # Update posterior
                "deter_state": cur_hidden_state["post"]["deter_state"].repeat(n,1,1) if cur_hidden_state["post"]["deter_state"] is not None else None, # (B,K,R)
                "lambdas1": cur_hidden_state["post"]["lambdas1"].repeat(n,1,1),  # (n,K,R)
                "lambdas2": cur_hidden_state["post"]["lambdas2"].repeat(n,1,1),  # (n,K,R)
                "extra_info": tmp,  # Each (B*K,R2)
                "samples": cur_hidden_state["post"]["samples"].repeat(n,1,1)  # (n,K,R)
            }
        }
        return new_hidden_state

    def select_specific_state(self, cur_hidden_state, indxs):
        tmp = [cur_hidden_state["post"]["extra_info"][0][indxs],
               cur_hidden_state["post"]["extra_info"][1][indxs]]

        new_hidden_state = {
            "prior": {
                "lambdas1": cur_hidden_state["prior"]["lambdas1"][indxs],  # (n,K,R)
                "lambdas2": cur_hidden_state["prior"]["lambdas1"][indxs],  # (n,K,R)
            },
            "post": {  # Update posterior
                "deter_state": cur_hidden_state["post"]["deter_state"][indxs] if
                cur_hidden_state["post"][ "deter_state"] is not None else None, # (B,K,R)
                "lambdas1": cur_hidden_state["post"]["lambdas1"][indxs],  # (n,K,R)
                "lambdas2": cur_hidden_state["post"]["lambdas2"][indxs],  # (n,K,R)
                "extra_info": tmp,  # Each (B*K,R2)
                "samples": cur_hidden_state["post"]["samples"][indxs]  # (n,K,R)
            }
        }
        return new_hidden_state


    # Input: array_of_states
    # Output: One state containing the information of the previous states
    def _stack_state(self, array_of_states):
        new_hidden_state = {
            "prior": {
                "lambdas1": [],  # (n,K,R)
                "lambdas2": [],  # (n,K,R)
            },
            "post": {  # Update posterior
                "deter_state": [],
                "lambdas1": [],  # (n,K,R)
                "lambdas2": [],  # (n,K,R)
                "extra_info_0": [],  # (n*K,R2)
                "extra_info_1": [],  # (n*K,R2)
                "samples": []  # (n,K,R)
            }
        }

        for a_state in array_of_states:
            new_hidden_state["prior"]["lambdas1"].append(a_state["prior"]["lambdas1"])
            new_hidden_state["prior"]["lambdas2"].append(a_state["prior"]["lambdas2"])

            new_hidden_state["post"]["deter_state"].append(a_state["post"]["deter_state"])
            new_hidden_state["post"]["lambdas1"].append(a_state["post"]["lambdas1"])
            new_hidden_state["post"]["lambdas2"].append(a_state["post"]["lambdas2"])
            new_hidden_state["post"]["extra_info_0"].append(a_state["post"]["extra_info"][0])
            new_hidden_state["post"]["extra_info_1"].append(a_state["post"]["extra_info"][1])
            new_hidden_state["post"]["samples"].append(a_state["post"]["samples"])

        new_hidden_state["prior"]["lambdas1"] = torch.cat(new_hidden_state["prior"]["lambdas1"])
        new_hidden_state["prior"]["lambdas2"] = torch.cat(new_hidden_state["prior"]["lambdas2"])

        if new_hidden_state["post"]["deter_state"][0] is not None:
            new_hidden_state["post"]["deter_state"] = torch.cat(new_hidden_state["post"]["deter_state"])
        else:
            new_hidden_state["post"]["deter_state"] = None
        new_hidden_state["post"]["lambdas1"] = torch.cat(new_hidden_state["post"]["lambdas1"])
        new_hidden_state["post"]["lambdas2"] = torch.cat(new_hidden_state["post"]["lambdas2"])
        new_hidden_state["post"]["extra_info"] = [torch.cat(new_hidden_state["post"]["extra_info_0"]),
                                                  torch.cat(new_hidden_state["post"]["extra_info_1"])]
        new_hidden_state["post"]["samples"] = torch.cat(new_hidden_state["post"]["samples"])
        return new_hidden_state

    # Description: Runs inference with the given inputs and returns a state_info dictionary
    # Inputs: obs (B,T1,3,D,D) or None, actions (B,T2,A) or None, initial_hidden_state or None, schedule (T3), select_best=False
    #   Note: Assume that initial_hidden_state has entries of size (B=1,*)
    # Outputs:
    #   If select_best==True: We assume that we are not doing a rollout and that our final output should be
    #       the same as obs[:,-1]. We therefore pick the state_info that results in the best output
    #       (e.g. lowest reconstruction error) so output B=1
    #   Else:
    #       Output B remains the same
    def batch_internal_inference(self, obs, actions, initial_hidden_state, schedule, select_best=False, figure_path=None, batch_size=10):
        loss_schedule = np.zeros_like(schedule)
        if obs is not None:
            B = obs.shape[0]
        elif actions is not None:
            B = actions.shape[0]
        else:
            raise ValueError("Unknown size of inputs!")

        num_batches = int(np.ceil(B/batch_size))

        important_info = {
            "colors": [],
            "masks": [],
            "sub_images": [],
            "final_recon": [],
            "state": []
        }

        for i in range(num_batches):
            start_index = i*batch_size
            end_index = min(start_index+batch_size, B)

            if obs is not None:
                batch_obs = obs[start_index:end_index]  # (b,T1,3,D,D)
            else:
                batch_obs = None

            if actions is not None:
                batch_actions = actions[start_index:end_index]  # (b,T2,A)
            else:
                batch_actions = None

            if initial_hidden_state is None or initial_hidden_state["post"]["samples"].shape[0] == 1:
                batch_initial_hidden_state = self.replicate_state(initial_hidden_state, end_index-start_index)
            else:
                batch_initial_hidden_state = self.select_specific_state(initial_hidden_state, range(start_index, end_index))
            colors, masks, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state = \
                self.run_schedule(batch_obs, batch_actions, batch_initial_hidden_state, schedule=schedule,
                                  loss_schedule=loss_schedule, should_detach=True)

            important_info["colors"].append(colors[:, -1]) #(b,K,3,D,D)
            important_info["masks"].append(masks[:, -1]) #(b,K,1,D,D)
            important_info["sub_images"].append((colors[:, -1] * masks[:, -1])) #(b,K,3,D,D)
            important_info["final_recon"].append(final_recon) #(b,3,D,D)
            important_info["state"].append(cur_hidden_state)

            if i == 0 and figure_path is not None:
                # true_images (T1,3,D,D),  colors (T,K,3,D,D),  masks (T,K,1,D,D), schedule (T)
                # file_name (string),  quicksave_type is either "full" or "subimages"
                # Images are torch tensors, schedule is numpy array
                quicksave(obs[0], colors[0], masks[0], schedule, figure_path, "full")

        important_info["colors"] = torch.cat(important_info["colors"]) # (B,K,3,D,D)
        important_info["masks"] = torch.cat(important_info["masks"])  # (B,K,1,D,D)
        important_info["sub_images"] = torch.cat(important_info["sub_images"])  # (B,K,3,D,D)
        important_info["final_recon"] = torch.cat(important_info["final_recon"])  # (B,3,D,D)
        important_info["state"] = self._stack_state(important_info["state"]) # State with (B,*) entries

        if select_best:
            mses = torch.pow(important_info["final_recon"] - obs[:, -1], 2).mean((-1, -2, -3))  # (B,3,D,D) -> (B)
            best_index = torch.argmin(mses)

            for akey in important_info:
                if akey == "state":
                    important_info[akey] = self.select_specific_state(important_info[akey], [[best_index]])
                else:
                    important_info[akey] = important_info[akey][best_index:best_index + 1]

            if figure_path is not None:
                visualize_state_info(important_info, file_name="{}_best.{}".format(*figure_path.split(".")),
                                     true_image=obs[best_index, -1])

        return important_info

    # Input: initial_hidden_state,  actions (B,A)
    # Output: Inertial action values (B,K)
    def get_activation_values(self, initial_hidden_state, actions, batch_size=10):
        B = actions.shape[0]
        num_batches = int(np.ceil(B / batch_size))

        act_values = ptu.zeros((B, self.K), torch_device=actions.device)  # (B,K)

        for i in range(num_batches):
            start_index = i*batch_size
            end_index = min(start_index+batch_size, B)

            batch_initial_hidden_state = self.replicate_state(initial_hidden_state, end_index - start_index)  # (b,...)
            states = self.get_full_tensor_state(batch_initial_hidden_state)  # (b,K,R)
            states = self._flatten_first_two(states)  # (b*K,R)
            input_actions = actions[start_index:end_index].repeat(self.K, 1)  # (b*K,A)
            vals = self.dynamics_net.get_state_action_attention_values(states, input_actions)  # (b*k, 1)
            act_values[start_index:end_index] = self._unflatten_first(vals, self.K)[:, :, 0]  # (b,k)

        return act_values

    # Input: initial_hidden_state,  actions (B,A)
    # Output: state_action_attention (B,K),  interaction_attention (B,K,K-1)
    def get_all_activation_values(self, initial_hidden_state, actions, batch_size=10):
        B = actions.shape[0]
        num_batches = int(np.ceil(B / batch_size))

        state_action_attention = ptu.zeros((B, self.K), torch_device=actions.device)  # (B,K)
        interaction_attention = ptu.zeros((B, self.K, self.K-1), torch_device=actions.device)  # (B,K,K-1)
        all_delta_vals = ptu.zeros((B, self.K), torch_device=actions.device)  # (B,K)
        all_lambdas_deltas = ptu.zeros((B, self.K), torch_device=actions.device)  # (B,K)

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, B)

            batch_initial_hidden_state = self.replicate_state(initial_hidden_state, end_index - start_index)  # (b,...)
            states = self.get_full_tensor_state(batch_initial_hidden_state)  # (b,K,R)
            states = self._flatten_first_two(states)  # (b*K,R)
            input_actions = actions[start_index:end_index].unsqueeze(1).repeat(1, self.K, 1)  # (b,K,A)
            input_actions = self._flatten_first_two(input_actions)
            state_vals, inter_vals, delta_vals = self.dynamics_net.get_all_attention_values(states, input_actions, self.K)  # (b*k,1), (b*k,k-1,1), (b*k,1)
            state_action_attention[start_index:end_index] = self._unflatten_first(state_vals, self.K)[..., 0]  # (b,k)
            interaction_attention[start_index:end_index] = self._unflatten_first(inter_vals, self.K)[..., 0]  # (b,k,k-1)
            all_delta_vals[start_index:end_index] = self._unflatten_first(delta_vals, self.K)[..., 0]  # (b,k)

            deter_state, lambdas1, lambdas2 = self.dynamics_net(states, input_actions)  # (b*k,Rd),  (b*k,Rs),  (b*k,Rs)
            lambdas_deltas = self._flatten_first_two(batch_initial_hidden_state["post"]["lambdas1"])  # (b,k,Rs)->(b*k,Rs)
            lambdas_deltas = torch.abs(lambdas_deltas - lambdas1).sum(1)  # (b*k,Rs)->(b*k)
            if deter_state is not None:
                deter_state_deltas = torch.abs(states[:, :self.det_size] - deter_state).sum(1)  # (b*k,Rd)->(b*k)
                lambdas_deltas += deter_state_deltas
            all_lambdas_deltas[start_index:end_index] = self._unflatten_first(lambdas_deltas, self.K)  # (b,k)

        return state_action_attention.detach(), interaction_attention.detach(), all_delta_vals.detach(), all_lambdas_deltas.detach()














