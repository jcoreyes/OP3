from op3.launchers.launcher_util import run_experiment

import op3.torch.op3_modules.op3_model as op3_model
from op3.torch.op3_modules.op3_trainer import TrainingScheduler, OP3Trainer


import numpy as np
import h5py
from op3.torch.data_management.dataset import Dataset, BlocksDataset
from torch.utils.data.dataset import TensorDataset
import torch
import random
from argparse import ArgumentParser
from op3.util.misc import get_module_path


def load_dataset(data_path, train=True, size=None, batchsize=8, static=True):
    try:
        hdf5_file = h5py.File(data_path, 'r')  # Data file
    except OSError:
        print("Dataset does not exist. Download dataset to %s" % data_path)
        exit(0)
    if 'stack' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features']) # (T, bs, ch, imsize, imsize)
        else:
            feats = np.array(hdf5_file['validation']['features'])
        feats = np.moveaxis(feats, -1, 2)  # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1)   # (bs, T, ch, imsize, imsize)
        torch_dataset = TensorDataset(torch.Tensor(feats)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        return dataset, T
    elif 'cloth' in data_path or 'pickplace' in data_path:
        #cloth: bs=13866, T=20, action_dim=4
        #pickplace_multienv_10k: bs=10000, T=21, action_dim=6
        if train:
            feats = np.array(hdf5_file['training']['features'])
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])
        feats = np.moveaxis(feats, -1, 2)  # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1)  # (bs, T, ch, imsize, imsize)
        actions = np.moveaxis(actions, 0, 1)  # (bs, T-1, action_dim) EXCEPT for pickplace envs which are (bs,T,A) instead
        if static:
            bs, T = feats.shape[0], feats.shape[1]
            if size == None:
                size = bs
            rand_ts = np.random.randint(0, T, size=size)  # As the first timesteps could be correlated
            tmp = torch.Tensor(feats[range(size), rand_ts]).unsqueeze(1)  # (size, 1, ch, imsize, imsize)
            torch_dataset = TensorDataset(tmp)
        else:
            torch_dataset = TensorDataset(torch.Tensor(feats[:size]), torch.Tensor(actions[:size]))
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)

        if 'pickplace' in data_path:
            dataset.action_dim = 4  # Changing it from 6 as we ignore the z values in the pickplace environment

        T = feats.shape[1]
        return dataset, T
    else:
        raise ValueError("Invalid dataset given: {}".format(data_path))


def train_vae(variant):
    from op3.core import logger
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ######Dataset loading######
    train_path = get_module_path() + '/data/datasets/{}.h5'.format(variant['dataset'])
    test_path = train_path
    bs = variant['training_args']['batch_size']
    train_size = 100 if variant['debug'] == 1 else None

    static = (variant['schedule_args']['schedule_type'] == 'static_iodine')  # Boolean
    train_dataset, max_T = load_dataset(train_path, train=True, batchsize=bs, size=train_size, static=static)
    test_dataset, _ = load_dataset(test_path, train=False, batchsize=bs, size=100, static=static)
    print(logger.get_snapshot_dir())

    ######Model loading######
    op3_args = variant["op3_args"]
    m = op3_model.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=train_dataset.action_dim)
    if variant['dataparallel']:
        m = torch.nn.DataParallel(m)
    m.cuda()

    ######Training######
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = max_T)
    t = OP3Trainer(train_dataset, test_dataset, m, scheduler, **variant["training_args"])

    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        train_stats = t.train_epoch(epoch)
        test_stats = t.test_epoch(epoch, train=False, batches=1, save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs)
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        t.save_model()


#Example run:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_op3.py -de 1 -va pickplace
if __name__ == "__main__":
    from op3.exp_variants.variants import *
    parser = ArgumentParser()
    parser.add_argument('-va', '--variant', type=str, required=True,
                        choices=['stack', 'pickplace', 'cloth'])
    parser.add_argument('-de', '--debug', type=int, default=1)  # Note: Change this to 0 to run on the entire dataset!
    parser.add_argument('-m', '--mode', type=str, default='here_no_doodad')  # Relevant options: 'here_no_doodad', 'local_docker', 'ec2'

    args = parser.parse_args()

    if args.variant == 'stack':
        variant = stack_variant
    elif args.variant == 'pickplace':
        variant = pickplace_variant
    elif args.variant == 'cloth':
        variant = cloth_variant
    else:
        raise Exception("Exp variant not found")

    variant['debug'] = args.debug
    run_experiment(
        train_vae,
        exp_prefix='{}'.format(args.variant),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
        region='us-west-2'  # Only used if mode is ec2
    )





