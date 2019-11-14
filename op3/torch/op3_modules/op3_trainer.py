from collections import OrderedDict
from collections import defaultdict
from os import path as osp
import numpy as np
import torch
from torch import optim
from op3.core import logger
from op3.core.serializable import Serializable
from op3.torch import pytorch_util as ptu
from op3.torch.pytorch_util import from_numpy

from op3.torch.op3_modules.visualizer import quicksave
import os
import pdb
import time


#######
class TrainingScheduler:
    #Seed_steps (Sc),  schedule_type (String),
    # max_T (Sc) denoting max number of frames in dataset.
    #   This is used to ensure that the schedule does not accidentally cause too many physics steps
    #T: Length of schedule for static schedule types
    def __init__(self, seed_steps, schedule_type, max_T, T):
        self.seed_steps = seed_steps
        self.schedule_type = schedule_type
        self.max_T = max_T
        self.T = T

        self.curriculum_offset = 100  #Used for "curriculum" schedule
        self.curriculum_increase = 10 #Used for "curriculum" schedule
        self.rprp_curriculum_len = 50 #Used for "rprp_curriculum" schedule

    #Input: epoch (Sc),  is_train (bool)
    #Output: schedule (T1) consisting of 0's, 1's, and 2's
    def get_schedule(self, epoch, is_train):
        schedule_type = self.schedule_type
        T = self.T
        seed_steps = self.seed_steps

        if schedule_type == 'single_step_physics':
            schedule = np.ones((T,))
            schedule[:seed_steps] = 0
        elif schedule_type == 'random_alternating':
            if is_train:
                schedule = np.random.randint(0, 2, (T,))
            else:
                schedule = np.ones((T,))
            schedule[:seed_steps] = 0
        elif schedule_type == 'multi_step_physics':
            schedule = np.ones((T,))
            schedule[:seed_steps] = 0
        elif schedule_type == 'curriculum':
            if epoch > self.curriculum_offset:
                max_multi_step = (epoch-self.curriculum_offset) // self.curriculum_increase + 1
            else:
                max_multi_step = 1
            max_multi_step = min(max_multi_step, T)
            if is_train:
                # max_multi_step = epoch//self.curriculum_len + 1
                rollout_len = np.random.randint(max_multi_step) + 1  # Enforces at least 1 physics step
                schedule = np.zeros(seed_steps + rollout_len + 1)
                schedule[seed_steps:seed_steps + rollout_len] = 1  # schedule looks like [0,0,0,0,1,1,1,0]
            else:
                # max_multi_step = epoch // self.curriculum_len + 1
                schedule = np.zeros(seed_steps + max_multi_step + 1)
                schedule[seed_steps:seed_steps + max_multi_step] = 1
        elif schedule_type == 'rprp_curriculum':
            if is_train:
                max_multi_step = epoch // self.rprp_curriculum_len + 1
                rollout_len = np.random.randint(max_multi_step) + 1
                schedule = np.zeros(seed_steps + rollout_len + 1 + rollout_len + 1)  # rrrrprpr
                schedule[seed_steps:] = 1
                schedule[seed_steps - 1::rollout_len + 1] = 0
            else:
                rollout_len = epoch // self.rprp_curriculum_len + 1
                schedule = np.zeros(seed_steps + rollout_len + 1 + rollout_len + 1)  # rrrrprpr
                schedule[seed_steps:] = 1
                schedule[seed_steps - 1::rollout_len + 1] = 0
        elif schedule_type == 'static_iodine':
            schedule = np.zeros((T,))
        elif schedule_type == 'rprp':
            schedule = np.zeros(seed_steps + (T - 1) * 2)
            schedule[seed_steps::2] = 1
        elif schedule_type == 'next_step':
            schedule = np.ones(T) * 2
            return schedule
        else:
            raise ValueError("{}".format(self.schedule_type))
        if self.max_T is not None:  # Enforces that we have at most max_T-1 physics steps
            timestep_count = np.cumsum(schedule)
            schedule = np.where(timestep_count <= self.max_T - 1, schedule, 0)
            # print(schedule)
        return schedule

    #Input: schedule (T1)
    #Output: loss_schedule (T1) with loss weights
    def get_loss_schedule(self, schedule):
        return np.arange(1, len(schedule)+1)

    #Input: epoch (Sc)
    #Output: If we should not save the model, output is None
    # If we should save the model, output is file_name (string)
    def should_save_model(self, epoch):
        if self.schedule_type == "curriculum":
            if epoch >= self.curriculum_offset and (epoch-self.curriculum_offset) % self.curriculum_increase == 0\
                    and (epoch-self.curriculum_offset)//self.curriculum_increase <= self.T:
                return "{}_physics_steps".format((epoch-self.curriculum_offset)//self.curriculum_increase)
        elif epoch % 50 == 49:
            return "{}_epoch_save".format(epoch)
        return None



#######Iodine trainer
class OP3Trainer(Serializable):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            scheduler_class,
            batch_size, #Training args
            lr=1e-3,
    ):
        self.quick_init(locals())
        self.batch_size = batch_size

        model.to(ptu.device)

        self.model = model
        self.scheduler_class = scheduler_class

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset

        self.batch_size = batch_size

        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None
        self.timing_info = defaultdict(list)

    def save_model(self, prefix=""):
        torch.save(self.model.state_dict(), open(osp.join(logger.get_snapshot_dir(), '{}_params.pkl'.format(prefix)), "wb"))

    def prepare_tensors(self, tensors):
        imgs = tensors[0].to(ptu.device) / 255. #Normalize image to 0-1
        if len(tensors) == 2:
            return imgs, tensors[1].to(ptu.device)
        else:
            return imgs, None #Action is none

    def train_epoch(self, epoch):
        t_start = time.time()
        timings = []
        if self.scheduler_class.should_save_model(epoch): #We can save the model at intermediate steps
            self.save_model(self.scheduler_class.should_save_model(epoch))

        self.model.train()
        losses, log_probs, kles, mses = [], [], [], []
        # with torch.autograd.detect_anomaly():
        for batch_idx, tensors in enumerate(self.train_dataset.dataloader):
            # print(batch_idx)
            true_images, actions = self.prepare_tensors(tensors) #(B,T,3,D,D),  (B,T,A) or None
            self.optimizer.zero_grad()

            schedule = self.scheduler_class.get_schedule(epoch, is_train=True)
            loss_schedule = self.scheduler_class.get_loss_schedule(schedule)

            t0 = time.time()
            colors, masks, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state = \
                self.model.forward(true_images, actions, initial_hidden_state=None, schedule=schedule, loss_schedule=loss_schedule)
            t1 = time.time()

            #For DataParallel
            total_loss = total_loss.mean()
            total_clog_prob = total_clog_prob.mean()
            total_kle_loss = total_kle_loss.mean()
            mse = mse.mean()

            # ptu.check_nan([total_loss], folder=logger.get_snapshot_dir())
            total_loss.backward()
            # ptu.check_nan([x.data for x in self.model.parameters()], folder=logger.get_snapshot_dir())

            t2 = time.time()
            torch.nn.utils.clip_grad_norm_([x for x in self.model.parameters()], 5.0)
            self.optimizer.step()
            # ptu.check_nan([x.data for x in self.model.parameters()])
            t3 = time.time()
            timings.append([t0, t1, t2, t3])

            losses.append(total_loss.item())
            log_probs.append(total_clog_prob.item())
            kles.append(total_kle_loss.item())
            mses.append(mse.item())

        timings = np.array(timings)
        difs = timings[:, 1:] - timings[:, :-1]
        difs = np.sum(difs, axis=0)
        print(difs)

        t_end = time.time()

        stats = OrderedDict([
            ("train/epoch", epoch),
            ("train/Log Prob", np.mean(log_probs)),
            ("train/KL", np.mean(kles)),
            ("train/loss", np.mean(losses)),
            ("train/mse", np.mean(mses)),
            ("train/time_forward", difs[0]),
            ("train/time_backwards", difs[1]),
            ("train/time_opt_step", difs[2]),
            ("train/time_total", t_end-t_start)
        ])
        return stats


    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            train=True,
            batches=1,
    ):

        schedule = self.scheduler_class.get_schedule(epoch, is_train=True)
        loss_schedule = self.scheduler_class.get_loss_schedule(schedule)

        self.model.eval()
        losses, log_probs, kles, mses = [], [], [], []
        dataloader = self.train_dataset.dataloader if train else self.test_dataset.dataloader
        for batch_idx, tensors in enumerate(dataloader):
            true_images, actions = self.prepare_tensors(tensors) #(B,T,3,D,D),  (B,T,A) or None

            colors, masks, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state = \
                self.model.forward(true_images, actions, initial_hidden_state=None, schedule=schedule,
                                        loss_schedule=loss_schedule)

            # For DataParallel
            total_loss = total_loss.mean()
            total_clog_prob = total_clog_prob.mean()
            total_kle_loss = total_kle_loss.mean()
            mse = mse.mean()

            losses.append(total_loss.item())
            log_probs.append(total_clog_prob.item())
            kles.append(total_kle_loss.item())
            mses.append(mse.mean().item())

            # if batch_idx == 0 and save_reconstruction:
            #     quicksave(true_images[0], colors[:,0], masks[:,0], schedule=schedule,
            #               file_name=logger.get_snapshot_dir()+"/{}_{}.png".format('train' if train else 'val', epoch), quicksave_type="full")
            if batch_idx == 0 and save_reconstruction:
                quicksave(true_images[0], colors[0], masks[0], schedule=schedule,
                          file_name=logger.get_snapshot_dir()+"/{}_{}.png".format('train' if train else 'val', epoch), quicksave_type="full")


            if batch_idx >= batches - 1:
                break


        stats = OrderedDict([
            ("test/Log Prob", np.mean(log_probs)),
            ("test/KL", np.mean(kles)),
            ("test/loss", np.mean(losses)),
            ("test/mse", np.mean(mses))
        ])

        return stats