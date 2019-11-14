import os
import argparse
import pickle
import random
import pdb
import sys
import h5py

# sys.path.append("..")
from XML import XML
from logger import Logger
import contacts
import utils
import numpy as np
import imageio
import cv2
import pathos.pools as pp


parser = argparse.ArgumentParser()
## stuff you might want to edit
parser.add_argument('--start', default=0, type=int, 
        help='starting index (useful if rendering in parallel jobs)')
parser.add_argument('--num_images', default=5, type=int,
        help='total number of images to render')
parser.add_argument('--img_dim', default=64, type=int,
        help='image dimension')
parser.add_argument('--output_path', default='../rendered/', type=str,
        help='path to save images')

parser.add_argument('--drop_steps_max', default=500, type=int,
        help='max number of steps simulating dropped object')
parser.add_argument('--render_freq', default=25, type=int,
        help='frequency of image saves in drop simulation')

parser.add_argument('--min_objects', default=4, type=int,
        help='min number of objects *starting on the ground*')
parser.add_argument('--max_objects', default=4, type=int,
        help='max number of objects *starting on the ground*')

## stuff you probably don't need to edit
parser.add_argument('--settle_steps_min', default=2000, type=int,
        help='min number of steps simulating ground objects to rest')
parser.add_argument('--settle_steps_max', default=2000, type=int,
        help='max number of steps simulating ground objects to rest')
parser.add_argument('--save_images', default=True, type=bool,
        help='if true, saves images as png (alongside pickle files)')
parser.add_argument('--filename', default="blocks.h5", type=str,   #RV: Added
        help='Name of the file. Please include .h5 at the end.')
parser.add_argument('-p', '--num_workers', type=int, default=1)
args = parser.parse_args()


polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']

num_objects = range(args.min_objects, args.max_objects + 1)

## bounds for objects that start on the ground plane
settle_bounds = {  
            'pos':   [ [-.5, .5], [-.5, 0], [1, 2] ],
            'hsv': [ [0, 1], [0.5, 1], [0.5, 1] ],
            'scale': [ [0.4, 0.4] ],
            'force': [ [0, 0], [0, 0], [0, 0] ]
          }

## bounds for the object to be dropped
drop_bounds = {  
            'pos':   [ [-1.75, 1.75], [-.5, 0], [0, 3] ],
          }  

## folder with object meshes
asset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mujoco_data/stl/')

#utils.mkdir(args.output_path)

metadata = {'polygons': polygons, 'max_steps': args.drop_steps_max, 
            'min_objects': min(num_objects), 
            'max_objects': max(num_objects)}
#pickle.dump( metadata, open(os.path.join(args.output_path, 'metadata.p'), 'wb') )

num_images_per_scene =  2 #int(args.drop_steps_max / args.render_freq)

end = args.start + args.num_images
# for img_num in tqdm.tqdm( range(args.start, end) ):
#
#     sim, xml, drop_name = contacts.sample_settled(asset_path, num_objects, polygons, settle_bounds)
#     logger = Logger(xml, sim, steps = num_images_per_scene, img_dim = args.img_dim )
#
#     ## drop all objects except [ drop_name ]
#     logger.settle_sim(drop_name, args.settle_steps_min, args.settle_steps_max)
#
#     ## filter scenes in which objects are intersecting
#     ## because it makes the physics unpredictable
#     overlapping = True
#     while overlapping:
#
#         ## get position for drop block
#         if random.uniform(0, 1) < 0.5 and len(xml.meshes) > 1:
#             ## some hard-coded messiness
#             ## to drop a block directly on top
#             ## of an existnig block half of the time
#             mesh = random.choice(xml.meshes)
#             pos = [float(p) for p in mesh['pos'].split(' ')]
#             pos[2] += random.uniform(.4, .8)
#         else:
#             ## drop on random position
#             pos = utils.uniform(*drop_bounds['pos'])
#
#         ## get orientation for drop block
#         if 'horizontal' in drop_name:
#             axangle = [1,0,0,0]
#         else:
#             axis = [0,0,1]
#             axangle  = utils.random_axangle(axis=axis)
#
#         ## position and orient the block
#         logger.position_body(drop_name, pos, axangle)
#
#         ## check whether there are any block intersections
#         overlapping = contacts.is_overlapping(sim, drop_name)
#
#     for i in range(args.drop_steps_max):
#         ## log every [ render_freq ] steps
#         if i % args.render_freq == 0:
#             logger.log(i//args.render_freq)
#         ## simulate one timestep
#         sim.step()
#
#     data, images, masks = logger.get_logs()
#
#     if args.save_images:
#         for timestep in range( images.shape[0] ):
#             plt.imsave( os.path.join(args.output_path, '{}_{}.png'.format(img_num, timestep)), images[timestep] / 255. )
#
#     config_path  = os.path.join( args.output_path, '{}.p'.format(img_num) )
#
#     config = {'data': data, 'images': images, 'masks': masks, 'drop_name': drop_name}
#
#     pickle.dump( config, open(config_path, 'wb') )


#####RV Modifications######
def action_to_vector(dict_with_info):
    num_type_polygons = len(polygons)
    total_size_of_array = num_type_polygons + 7 + 3 #7 for qpos, 3 for rgba
    ans = np.zeros(total_size_of_array)

    poly_name = dict_with_info['ply']
    val = polygons.index(poly_name)
    ans[val] = 1

    for i in range(len(dict_with_info["qpos"])):
        ans[num_type_polygons + i] = dict_with_info["qpos"][i]

    for i in range(len(dict_with_info["rgba"])):
        ans[num_type_polygons + 7 + i] = dict_with_info["rgba"][i]
    # print(ans)
    return ans


def createSingleSim(_):
    sim, xml, drop_name = contacts.sample_settled(asset_path, num_objects, polygons, settle_bounds)
    # print(drop_name) #RV: Drop name is a string ending with _number
    logger = Logger(xml, sim, steps=num_images_per_scene, img_dim=args.img_dim)

    ## drop all objects except [ drop_name ]
    logger.settle_sim(drop_name, args.settle_steps_min, args.settle_steps_max)
    logger.log(0)

    ## filter scenes in which objects are intersecting
    ## because it makes the physics unpredictable
    overlapping = True
    while overlapping:
        ## get position for drop block
        if random.uniform(0, 1) < 0.5 and len(xml.meshes) > 1:
            ## some hard-coded messiness
            ## to drop a block directly on top
            ## of an existnig block half of the time
            mesh = random.choice(xml.meshes)
            pos = [float(p) for p in mesh['pos'].split(' ')]
            pos[2] += random.uniform(.4, .8)
        else:
            ## drop on random position
            pos = utils.uniform(*drop_bounds['pos'])

        ## get orientation for drop block
        if 'horizontal' in drop_name:
            axangle = [1, 0, 0, 0]
        else:
            axis = [0, 0, 1]
            axangle = utils.random_axangle(axis=axis)

        ## position and orient the block
        logger.position_body(drop_name, pos, axangle)

        ## check whether there are any block intersections
        overlapping = contacts.is_overlapping(sim, drop_name)
    # print(logger.get_state()[drop_name])
    action_vec = action_to_vector(logger.get_state()[drop_name]) #RV addition

    # print(args.render_freq, args.drop_steps_max)
    logger.log(0)
    for i in range(args.drop_steps_max):
        ## log every [ render_freq ] steps
        #if i % args.render_freq == 0:
            ## print(i, (i // args.render_freq) + 1)
        #   logger.log((i // args.render_freq) + 1)
        ## simulate one timestep
        sim.step()
    logger.log(1)
    data, images, masks = logger.get_logs()

    return images, action_vec


filename = os.path.join(args.output_path, args.filename)
def createMultipleSims(num_workers=1):
    datasets = {'training':args.num_images,'validation':min(args.num_images, 100)}
    n_frames =   2 #(args.drop_steps_max-1) // args.render_freq + 1 + 1 #Default: 500//25=20. Make
    # it 500//15 = 33
    image_res = args.img_dim
    pool = pp.ProcessPool(num_workers)
    import pdb; pdb.set_trace()
    with h5py.File(filename, 'w') as f:
        for folder in datasets:
            cur_folder = f.create_group(folder)

            num_sims = datasets[folder]

            # create datasets, write to disk
            image_data_shape = (n_frames, num_sims, image_res, image_res, 3)
            #groups_data_shape = (n_frames, num_sims, image_res, image_res, 1)
            action_data_shape = (1, num_sims, len(polygons) + 7 + 3)
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='uint8')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')

            results = pool.map(createSingleSim, [(0) for _ in range(num_sims)])
            for i in range(num_sims):
                frames, action_vec = results[i] #(T, M, N, C), (T, M, N, 1)

                # RV: frames is numpy with shape (T, M, N, C)
                # Bouncing balls dataset has shape (T, 1, M, N, C)
                #frames = np.expand_dims(frames, 1)

                features_dataset[:, i, :, :, :] = frames
                action_dataset[0, i, :] = action_vec

            print("Done with dataset: {}".format(folder))


def make_gif(images_root, gifname):
    file_names = [fn for fn in os.listdir(images_root) if fn.endswith('.png')]
    file_names =  sorted(file_names, key=lambda x: int(os.path.splitext(x)[0]))
    # images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    images = []
    for a_file in file_names:
        images.append(imageio.imread(os.path.join(images_root, a_file)))
    imageio.mimsave(images_root + gifname, images)

def mkdirp(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def hdf5_to_image(filename):
    root = os.path.dirname(filename)
    img_root = mkdirp(os.path.join(root, 'imgs'))
    h5file = h5py.File(filename, 'r')
    for mode in h5file.keys():
        mode_folder = mkdirp(os.path.join(img_root, mode))
        groups = h5file[mode]
        f = groups['features']
        for ex in range(f.shape[1]):
            ex_folder = mkdirp(os.path.join(mode_folder, str(ex)))
            for d in groups.keys():
                if d in ['features', 'groups', 'collisions']:
                    dataset_folder = mkdirp(os.path.join(ex_folder, d))
                    dataset = groups[d]
                    num_groups = np.max(dataset[:, ex])
                    for j in range(dataset.shape[0]):
                        imfile = os.path.join(dataset_folder, str(j)+'.png')
                        if d == 'features':
                            cv2.imwrite(imfile, dataset[j, ex]*255)
                        elif d == 'groups':
                            cv2.imwrite(imfile, dataset[j, ex]*255.0/(num_groups))
                        elif d == 'collisions':
                            cv2.imwrite(imfile, dataset[j, ex]*255)
                        else:
                            assert False


createMultipleSims(int(args.num_workers))
print("Done with creating the dataset, now creating visuals")
hdf5_to_image(filename)
for i in range(5):
    tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
    make_gif(tmp, "animation.gif")
    # tmp = os.path.join(args.output_path, "imgs/training/{}/groups".format(str(i)))
    # make_gif(tmp, "animation.gif")




