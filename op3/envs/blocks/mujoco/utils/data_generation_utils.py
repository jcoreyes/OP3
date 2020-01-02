import os
import argparse
import pickle
import random
import pdb
import sys
import h5py

# sys.path.append("..")
import numpy as np
import imageio
import cv2
import pathos.pools as pp

def createMultipleSims(args, obs_size, ac_size, createSingleSim, num_workers=1):
    datasets = {'training': args.num_sims, 'validation': min(args.num_sims, 100)}
    n_frames = args.num_frames
    all_results = []
    for folder in datasets:
        num_sims = datasets[folder]
        if num_workers == 1:
            results = [createSingleSim(args) for _ in range(num_sims)]
        else:
            pool = pp.ProcessPool(num_workers)
            results = pool.map(createSingleSim, [args for _ in range(num_sims)])
        all_results.append(results)

    # Save env data to pickle
    if 'env' in all_results[0][0].keys():
        env_data = {}
        for f_idx, folder in enumerate(datasets):
            env_data[folder] = [a_result['env'] for a_result in all_results[f_idx]]
        with open(args.filename+'.pkl', 'wb') as f:
            pickle.dump(env_data, f)


    # Save data to hdf5 file
    with h5py.File(args.filename+'.h5', 'w') as f:
        for f_idx, folder in enumerate(datasets):
            num_sims = datasets[folder]
            cur_folder = f.create_group(folder)
            # create datasets, write to disk
            # image_data_shape = (n_frames, num_sims, image_res, image_res, 3)
            image_data_shape = [n_frames, num_sims] + obs_size + [3]
            # groups_data_shape = [n_frames, num_sims] + list(env.get_obs_size()) + [1]
            # action_data_shape = (1, num_sims, len(polygons) + 7 + 3)
            action_data_shape = [n_frames, num_sims] + ac_size
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='uint8')
            # groups_dataset = cur_folder.create_dataset('groups', groups_data_shape, dtype='float32')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')

            if 'segs' in all_results[0][0]:
                segs_data_shape = [n_frames, num_sims, 2] + obs_size + [3] #(T,B,segs=2,D,D,3)
                segs_dataset = cur_folder.create_dataset('segs', segs_data_shape, dtype='float32')

            for i in range(num_sims):
                # frames, action_vec = all_results[f_idx][i] #createSingleSim()  # (T, M, N, C), (T, A)
                a_result_dict = all_results[f_idx][i]
                frames = a_result_dict['features']
                action_vec = a_result_dict['actions']

                if 'segs' in a_result_dict:
                    segs = a_result_dict['segs']
                    segs_dataset[:, i] = segs

                # frames, group_frames, action_vec = createSingleSim() #(T, M, N, C), (T, M, N, 1)
                # pdb.set_trace()

                # RV: frames is numpy with shape (T, M, N, C)
                # Bouncing balls dataset has shape (T, 1, M, N, C)
                #frames = np.expand_dims(frames, 1)
                # group_frames = np.expand_dims(group_frames, 1)

                features_dataset[:, i, :, :, :] = frames
                # groups_dataset[:, [i], :, :, :] = group_frames
                action_dataset[:, i, :] = action_vec
            #import pdb; pdb.set_trace()
            print("Done with dataset: {}".format(folder))


def make_gif(images_root, gifname):
    file_names = [fn for fn in os.listdir(images_root) if fn.endswith('.png')]
    file_names =  sorted(file_names, key=lambda x: int(os.path.splitext(x)[0]))
    # images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    images = []
    for a_file in file_names:
        images.append(imageio.imread(os.path.join(images_root,a_file)))
    imageio.mimsave(images_root + gifname, images)

def mkdirp(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def hdf5_to_image(filename, num_to_visualize=100):
    root = os.path.dirname(filename)
    img_root = mkdirp(os.path.join(root, 'imgs'))
    h5file = h5py.File(filename, 'r')
    for mode in h5file.keys():
        mode_folder = mkdirp(os.path.join(img_root, mode))
        groups = h5file[mode]
        f = groups['features']
        for ex in range(min(f.shape[1], num_to_visualize)):
            ex_folder = mkdirp(os.path.join(mode_folder, str(ex)))
            for d in groups.keys():
                if d in ['features', 'groups', 'collisions']:
                    dataset_folder = mkdirp(os.path.join(ex_folder, d))
                    dataset = groups[d]
                    num_groups = np.max(dataset[:, ex])
                    for j in range(dataset.shape[0]):
                        imfile = os.path.join(dataset_folder, str(j)+'.png')
                        if d == 'features':
                            if (np.max(dataset[j, ex]) > 1):
                                cv2.imwrite(imfile, dataset[j, ex])
                            else:
                                cv2.imwrite(imfile, dataset[j, ex]*255)
                        elif d == 'groups':
                            cv2.imwrite(imfile, dataset[j, ex]*255.0/(num_groups))
                        elif d == 'collisions':
                            cv2.imwrite(imfile, dataset[j, ex]*255)
                        else:
                            assert False
            if ex > 20:
                break


