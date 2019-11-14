import os
import argparse
import pickle
import random
import copy
import pdb
import time

import colorsys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import mujoco_py as mjc

from op3.envs.blocks.mujoco.logger import Logger
import op3.envs.blocks.mujoco.utils as utils
from op3.envs.blocks.mujoco.XML import XML


class BlockEnv:
    def __init__(self, max_num_objects_dropped):
        self.asset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mujoco_data/stl/')
        self.img_dim = 64
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']
        self.settle_bounds = {
            'pos':   [ [-.5, .5], [-.5, 0], [1, 2] ],
            'hsv': [ [0, 1], [0.5, 1], [0.5, 1] ],
            'scale': [ [0.4, 0.4] ],
            'force': [ [0, 0], [0, 0], [0, 0] ]
          }

        self.drop_bounds = {
            'pos':   [ [-1.75, 1.75], [-.5, 0], [0, 3] ],
          }

        self.xml = XML(self.asset_path)

        xml_str = self.xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        self.max_num_objects_dropped = max_num_objects_dropped
        self.logger = Logger(self.xml, sim, steps=max_num_objects_dropped + 1, img_dim=self.img_dim)
        self.logger.log(0)

        self._blank_observation = self.get_observation()  # This can be / is used in the mpc goal acquisition step

        self.xml_actions_taken = []
        self.names = []
        self.env_step = 0
        self.settle_steps = 2000

    def reset(self):
        xml = XML(self.asset_path)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)
        self.logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        self.logger.log(0)

        self.xml_actions_taken = []
        self.names = []
        self.env_step = 0

        return self.get_observation()


    def get_observation(self):
        data, images, masks = self.logger.get_logs()
        image = images[0] #/255
        return image

    def sample_action(self):
        ply = random.choice(self.polygons)

        pos = utils.uniform(*self.drop_bounds['pos'])
        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle = utils.random_axangle(axis=axis)
        # axangle[-1] = 0 #Uncomment to remove rotation

        scale = utils.uniform(*self.settle_bounds['scale'])  # Note: Scale is ignored in self.xml_action_to_model_action
        rgba = self.sample_rgba_from_hsv(*self.settle_bounds['hsv'])
        xml_action = {
            'polygon': ply,
            'pos': pos,
            'axangle': axangle,
            'scale': scale,
            'rgba': rgba
        }
        return self.xml_action_to_model_action(xml_action)

    def sample_action_gaussian(self, mean, std):
        ply_t = .1
        ply_p = (mean[:3] + ply_t) / (mean[:3] + ply_t).sum()
        ply = np.random.choice(self.polygons, p=ply_p)

        std = np.maximum(std, 0.01)
        random_a = np.random.normal(mean, std)
        pos = np.clip(random_a[3:6], [x[0] for x in self.drop_bounds['pos']],
                      [x[1] for x in self.drop_bounds['pos']])
        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle = utils.random_axangle(axis=axis)
        axangle[-1] = random_a[9]
        # axangle[-1] = 0 #Uncomment to remove angle from action space

        if 'horizontal' in ply:
            axangle[-1] = 0

        scale = utils.uniform(*self.settle_bounds['scale']) # Note: Scale is ignored in self.xml_action_to_model_action
        # rgba = self.sample_rgba_from_hsv(*self.settle_bounds['hsv'])
        # rgba = np.clip(random_a[-3:], [x[0] for x in self.settle_bounds['hsv']],
        #               [x[1] for x in self.settle_bounds['hsv']])
        rgba = np.clip(random_a[-3:], 0, 1)

        xml_action = {
            'polygon': ply,
            'pos': pos,
            'axangle': axangle,
            'scale': scale,
            'rgba': rgba
        }
        return self.xml_action_to_model_action(xml_action)

    def sample_multiple_action_gaussian(self, mean, std, num_actions):
        return np.stack([self.sample_action_gaussian(mean, std) for _ in range(num_actions)])

    def get_obs_size(self):
        return (self.img_dim, self.img_dim)

    def get_actions_size(self):
        return (15)

    #Olld step
    # def step(self, an_action):
    #     #an_action should contain one_hot of polygon[3], pos[3], axangle[4], scale[1], rgba[3]
    #     #Total size: 3+3+4+1+4 = 15
    #
    #     xml = XML(self.asset_path)
    #     #Note: We need to recreate the entire scene
    #     for ind, prev_action in enumerate(self.xml_actions_taken): # Adding previous actions
    #         prev_action['pos'][-1] = ind*2
    #         xml.add_mesh(**prev_action)
    #
    #     xml_action = self.model_action_to_xml_action(an_action)
    #     # print("Action to take: ", xml_action)
    #     new_name = xml.add_mesh(**xml_action) #Note name is name of action (name of block dropped)
    #
    #     self.names.append(new_name)
    #
    #     xml_str = xml.instantiate()
    #     model = mjc.load_model_from_xml(xml_str)
    #     sim = mjc.MjSim(model)
    #
    #     logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
    #     # logger.log(0)
    #
    #     for act_ind, act in enumerate(self.names):
    #         logger.hold_drop_execute(self.names[act_ind+1:], self.names[act_ind], self.settle_steps)
    #         # logger.log(act_ind+1)
    #     # logger.hold_drop_execute(self.names, new_name, 1)
    #     # logger.log(len(self.xml_actions_taken)+1)
    #
    #     # print(self.xml_actions_taken)
    #     self.logger = logger
    #     self.logger.log(0)
    #
    #     ##Update state information
    #     self.xml_actions_taken.append(xml_action)
    #     # self.names.append(new_name)
    #
    #     return self.get_observation()

    def step(self, an_action):
        xml = XML(self.asset_path)
        # Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken):  # Adding previous actions
            prev_action['pos'][-1] = ind * 2
            xml.add_mesh(**prev_action)

        xml_action = self.model_action_to_xml_action(an_action)
        # print("Action to take: ", xml_action)
        new_name = xml.add_mesh(**xml_action)  # Note name is name of action (name of block dropped)

        self.names.append(new_name)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)

        #Set previous block states
        for a_block in self.names[:-1]:
            self.set_block_info(sim, a_block, self.get_block_info(a_block))

        logger.hold_drop_execute([], self.names[-1], self.settle_steps)
        self.logger = logger
        self.logger.log(0)

        ##Update state information
        self.xml_actions_taken.append(xml_action)
        self.sim = self.logger.sim

        return self.get_observation()

    # Tries an action and returns the direct observation of the action (e.g. the block in the air)
    #  but does not actually take a step in the environment
    def try_action(self, an_action):
        xml = XML(self.asset_path)
        # Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken):  # Adding previous actions
            prev_action['pos'][-1] = ind * 2
            xml.add_mesh(**prev_action)

        xml_action = self.model_action_to_xml_action(an_action)
        new_name = xml.add_mesh(**xml_action)  # Note name is name of action (name of block dropped)
        new_names = self.names + [new_name]

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        for a_block in self.names:
            self.set_block_info(sim, a_block, self.get_block_info(a_block))
        logger.hold_drop_execute([], new_name, 1)

        # for act_ind, act in enumerate(new_names[:-1]):
        #     logger.hold_drop_execute(new_names[act_ind+1:], new_names[act_ind], self.settle_steps)
        logger.log(0)

        original_logger = self.logger
        self.logger = logger
        obs = self.get_observation()
        self.logger = original_logger
        return obs

    def get_block_info(self, a_block):
        info = {}
        info["poly"] = a_block[:-2]
        info["pos"] = np.copy(self.logger.sim.data.get_body_xpos(a_block)) #np array
        info["quat"] = np.copy(self.logger.sim.data.get_body_xquat(a_block))
        info["vel"] = np.copy(self.logger.sim.data.get_body_xvelp(a_block))
        info["rot_vel"] = np.copy(self.logger.sim.data.get_body_xvelr(a_block))
        return info

    def set_block_info(self, sim, a_block, info):
        start_ind = sim.model.get_joint_qpos_addr(a_block)[0]
        sim_state = sim.get_state()
        if "pos" in info:
            sim_state.qpos[start_ind:start_ind+3] = np.array(info["pos"])
        if "quat" in info:
            sim_state.qpos[start_ind+3:start_ind+7] = info["quat"]
        else:
            sim_state.qpos[start_ind + 3:start_ind + 7] = np.array([1, 0, 0, 0])

        start_ind = sim.model.get_joint_qvel_addr(a_block)[0]
        if "vel" in info:
            sim_state.qvel[start_ind:start_ind + 3] = info["vel"]
        else:
            sim_state.qvel[start_ind:start_ind + 3] = np.zeros(3)
        if "rot_vel" in info:
            sim_state.qvel[start_ind + 3:start_ind + 6] = info["rot_vel"]
        else:
            sim_state.qvel[start_ind + 3:start_ind + 6] = np.zeros(3)
        sim.set_state(sim_state)



    #############Internal functions#########
    def model_action_to_xml_action(self, model_action):
        # an_action should contain one_hot of polygon[3], pos[3], axangle[4], scale[1], rgba[4]
        # Total size: 3+3+4+1+4 = 15, an_action should be of size [15]
        ans = {
            "polygon": self.polygons[np.where(model_action == 1)[0][0]],
            "pos": model_action[3:6],
            "axangle": model_action[6:10],
            "scale": 0.4,
            "rgba": np.concatenate([model_action[10:], np.array([1])])
        }
        return ans

    def xml_action_to_model_action(self, xml_action):
        num_type_polygons = len(self.polygons)
        total_size_of_array = 13  # num_type_polygons+3+4+1+4  #polygon[3], pos[3], axangle[4], scale[1], rgba[4]
        ans = np.zeros(total_size_of_array)

        poly_name = xml_action['polygon']
        val = self.polygons.index(poly_name)
        ans[val] = 1

        for i in range(len(xml_action["pos"])):
            ans[num_type_polygons + i] = xml_action["pos"][i]

        for i in range(len(xml_action["axangle"])):
            ans[num_type_polygons + 3 + i] = xml_action["axangle"][i]

        # ans[num_type_polygons + 3 + 4] = xml_action["scale"], we ignore scale

        for i in range(3):
            ans[num_type_polygons + 3 + 4 + i] = xml_action["rgba"][i]
        return ans

    def sample_rgba_from_hsv(self, *hsv_bounds):
        hsv = utils.uniform(*hsv_bounds)
        rgba = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return rgba

    def compute_accuracy(self, true_data):
        state = self.logger.get_state()
        return self.compare_matching(state, true_data['data']), state

    def compare_matching(self, data, mjc_data, threshold=0.2):
        # data is env, mjc_data is target
        # data = data.val[0].val
        mjc_data = copy.deepcopy(mjc_data)

        max_err = -float('inf')
        for pred_name, pred_datum in data.items():
            err, mjc_match, err_pos, err_rgb = self._best_obj_match(pred_datum, mjc_data)
            del mjc_data[mjc_match]

            if err > max_err:
                max_err = err
                max_pos = err_pos
                max_rgb = err_rgb

            if len(mjc_data) == 0:
                break

        correct = max_err < threshold
        return correct, max_pos, max_rgb

    def _best_obj_match(self, pred, targs):
        def np_mse(x1, x2):
            return np.square(x1 - x2).mean()

        pos = pred['qpos'][:3]
        rgb = pred['rgba']

        best_err = float('inf')
        for obj_name, obj_data in targs.items():
            obj_pos = obj_data['xpos'][-1]
            obj_rgb = obj_data['xrgba'][-1]

            pos_err = np_mse(pos, obj_pos)
            rgb_err = np_mse(rgb, obj_rgb)
            err = pos_err + rgb_err

            if err < best_err:
                best_err = err
                best_obj = obj_name
                best_pos = pos_err
                best_rgb = rgb_err

        return best_err, best_obj, best_pos, best_rgb


###########Start unit tests############
def sanity_check_accuracy():
    env1 = BlockEnv(5)
    actions = []
    for i in range(3):
        action = env1.sample_action()
        env1.step(action)
        actions.append(action)

    env2 = BlockEnv(5)
    for i in range(3):
        action = env2.sample_action_gaussian(np.array(actions[i]), 0.01)
        env2.step(action)

    true_data = env1.logger.get_state()
    for obj_name, obj_data in true_data.items():
        true_data[obj_name]['xpos'] = np.array([true_data[obj_name]['qpos'][:3]])
        true_data[obj_name]['xrgba'] = np.array([true_data[obj_name]['rgba'][:3]])
    true_data = {'data': true_data}
    tmp = env2.compute_accuracy(true_data)
    print(tmp)

    cur_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 6, 1 * 6))
    axes[0].imshow(env1.get_observation()/255, interpolation='nearest')
    axes[1].imshow(env2.get_observation()/255, interpolation='nearest')
    cur_fig.savefig("HELLO")


def check_bugs_in_try_action():
    num_blocks = 5
    env = BlockEnv(num_blocks)
    cur_fig, axes = plt.subplots(nrows=2, ncols=num_blocks, figsize=(num_blocks * 6, 2 * 6))
    for i in range(num_blocks):
        action = env.sample_action()
        axes[0, i].imshow(env.try_action(action)/255, interpolation='nearest')
        axes[1, i].imshow(env.step(action) / 255, interpolation='nearest')
    cur_fig.savefig("check_bugs_in_try_action")

def timing():
    num_actions = 100
    env = BlockEnv(4)
    t0 = time.time()
    actions = [env.sample_action() for _ in range(num_actions)]
    t1 = time.time()
    print("Sampling action time: {}".format(t1-t0))

    t0 = time.time()
    for i in range(num_actions):
        results = env.try_action(actions[i])
    t1 = time.time()
    print("Old try action: {}".format(t1-t0))

    t0 = time.time()
    for i in range(num_actions):
        results = env.try_action_2(actions[i])
    t1 = time.time()
    print("New try action: {}".format(t1 - t0))

def check_blank_observation():
    env = BlockEnv(4)
    bo = env._blank_observation
    plot_numpy(np.array([[bo, bo], [bo, bo]]), "check_blank_observation.png")


#Inputs: numpy_array (H,W,D,D,3)
def plot_numpy(numpy_array, file_name, titles=None):
    num_rows, num_cols = numpy_array.shape[:2]
    cur_fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_rows * 6, num_cols * 6))
    for y in range(num_rows):
        for x in range(num_cols):
            axes[y, x].imshow(numpy_array[y, x], interpolation='nearest')
            if titles is not None:
                axes[y, x].set_title(titles[y, x])
    cur_fig.savefig(file_name)
#############End unit tests######################

if __name__ == '__main__':
    check_blank_observation()
    # sanity_check_accuracy()
    # check_bugs_in_try_action()
    # timing()


