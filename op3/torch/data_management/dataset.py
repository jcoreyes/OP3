import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset
from op3.torch import pytorch_util as ptu
import h5py
import numpy as np


class Dataset:
    def __init__(self, torch_dataset, batchsize=8, shuffle=True):
        self.dataset = torch_dataset
        self.batchsize = batchsize
        self.shuffle = shuffle

        self._dataloader = DataLoader(torch_dataset, batch_size=batchsize, shuffle=shuffle,
                                      pin_memory=True)


    def set_batchsize(self, batchsize):
        self._dataloader = DataLoader(self.dataset, batch_size=batchsize, shuffle=self.shuffle)

    def add(self, dataset):
        self.dataset = ConcatDataset([self.dataset, dataset])
        self._dataloader = DataLoader(self.dataset, batch_size=self.batchsize, shuffle=self.shuffle)

    @property
    def dataloader(self):
        return self._dataloader


class BlocksDataset(Dataset):
    def __init__(self, torch_dataset, batchsize=8, shuffle=True):
        super().__init__(torch_dataset, batchsize, shuffle)
        if len(self.dataset.tensors) == 2:
            self.action_dim = self.dataset.tensors[1].shape[-1]
        else:
            self.action_dim = 0

    def __getitem__(self, idx):
        if self.action_dim == 0:
            frames = self._dataloader.dataset[idx][0] #(T, 3, D, D)
            return frames, None
        else:
            frames = self._dataloader.dataset[idx][0]  #(T, 3, D, D)
            actions = self._dataloader.dataset[idx][1] #(T-1, A) or (T, A)
            return frames, actions


class LazyDataset:
    def __init__(self, data_path, batchsize):
        self.data_path = data_path
        self.batchsize = batchsize
        if 'twoBalls' in data_path:
            self.train_dataset = h5py.File(data_path, 'r')['training']
            self.test_dataset = h5py.File(data_path, 'r')['validation']

    def get_batch(self, train):
        if 'twoBalls' in self.data_path:
            if train:
                n = self.train_dataset['features'].shape[1] # (51,1000,64,64,3), values btwn 0-1
                dataset = self.train_dataset
            else:
                n = self.test_dataset['features'].shape[1]
                dataset = self.train_dataset

            inds = np.random.randint(0, n, size=self.batchsize)

            batch = np.array(dataset['features'][:, inds]) * 255 #(51,B,64,64)
            batch = np.transpose(batch, (1, 0, 4, 2, 3))[:size, :8] * 255  # (B,8,3,64,46)
            return ptu.from_numpy(batch)