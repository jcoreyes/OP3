import os
import pdb
import numpy as np
import shutil
import pickle
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import op3.envs.blocks.mujoco.utils.data_generation_utils as dgu
from op3.util.plot import plot_multi_image
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import op3.envs.blocks.mujoco.contacts as contacts
import copy

MODEL_XML_BASE = """
<mujoco>
    <asset>
        <material name='wall_visible' rgba='.9 .9 .9 1' specular="0" shininess="0"  emission="0.25"/>
        <material name='wall_invisible' rgba='.9 .9 .9 0' specular="0" shininess="0" emission="0.25"/>
       {}
       {}
    </asset>
    <worldbody>
        <camera name='fixed' pos='0 -3 4.5' euler='-300 0 0' fovy='55'/>
        <light diffuse='1.5 1.5 1.5' pos='0 -7 8' dir='0 1 1'/>  
        <light diffuse='1.5 1.5 1.5' pos='0 -7 6' dir='0 1 1'/>  
        <geom name='wall_floor' type='plane' pos='0 0 0' euler='0 0 0' size='20 10 0.1' material='wall_visible' 
        condim='3' friction='1 1 1'/>
        <geom name='occluding_wall' type='box' pos='2 0 0' euler='0 0.3 0' size='2 0.1 5' material='wall_visible' />
        {}
    </worldbody>
</mujoco>
"""
#60 = -300
# <geom name='wall_front'  type='box' pos='0 -5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>
# <geom name='wall_left'  type='box' pos='-5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
# <geom name='wall_right'  type='box' pos='5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
# <geom name='wall_back'  type='box' pos='0 5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>

# <body name="floor" pos="0 0 0.025">
#     <geom size="3.0 3.0 0.02" rgba="0 1 0 1" type="box"/>
#     <camera name='fixed' pos='0 -8 8' euler='45 0 0'/>
# </body>

#Red, Lime, Blue, Yellow, Cyan, Magenta, Black, White
COLOR_LIST = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [1, 1, 1], [255, 255, 255]]
def pickRandomColor(an_int):
    if an_int is None:
        return np.random.uniform(low=0.0, high=1.0, size=3)
    tmp = np.random.randint(0, an_int)
    return np.array(COLOR_LIST[tmp])/255


class BlockPickAndPlaceEnv():
    def __init__(self, num_objects, num_colors, img_dim, include_z, random_initialize=False, view=False):
        # self.asset_path = os.path.join(os.getcwd(), '../mujoco_data/stl/')
        self.asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mujoco_data/stl/')
        # self.asset_path = os.path.join(os.path.realpath(__file__), 'mujoco_data/stl/')
        # self.asset_path = '../mujoco_data/stl/'
        self.img_dim = img_dim
        self.include_z = include_z
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron'][:1]
        self.num_colors = num_colors
        self.num_objects = num_objects
        self.view = view

        #Hyper-parameters
        self.internal_steps_per_step = 2000
        self.drop_height = 3
        self.pick_height = 0.59
        self.bounds = {'x_min': -2.5, 'x_max': 2.5, 'y_min': 1.0, 'y_max': 4.0, 'z_min': 0.05, 'z_max': 2.2}
        self.TOLERANCE = 0.2
        self.wall_pos = np.array([2, 0, 0])

        self.names = []
        self.blocks = []
        self._blank_observation = None

        if random_initialize:
            self.reset()

    ####Env initialization functions
    def get_unique_name(self, polygon):
        i = 0
        while '{}_{}'.format(polygon, i) in self.names:
            i += 1
        name = '{}_{}'.format(polygon, i)
        self.names.append(name)
        return name

    def add_mesh(self, polygon, pos, quat, rgba):
        name = self.get_unique_name(polygon)
        self.blocks.append({'name': name, 'polygon': polygon, 'pos': np.array(pos), 'quat': np.array(quat), 'rgba': rgba,
                            'material': name})

    def get_asset_material_str(self):
        asset_base = '<material name="{}" rgba="{}" specular="0" shininess="0" emission="0.25"/>'
        asset_list = [asset_base.format(a['name'], self.convert_to_str(a['rgba'])) for a in self.blocks]
        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_asset_mesh_str(self):
        asset_base = '<mesh name="{}" scale="0.6 0.6 0.6" file="{}"/>'
        asset_list = [asset_base.format(a['name'], os.path.join(self.asset_path, a['polygon'] + '.stl'))
                      for a in self.blocks]
        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_body_str(self):
        body_base = '''
          <body name='{}' pos='{}' quat='{}'>
            <joint type='free' name='{}'/>
            <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' 
            condim='3' friction='1 1 1' solimp="0.998 0.998 0.001" solref="0.02 1"/>
          </body>
        '''
        body_list = [body_base.format(m['name'], self.convert_to_str(m['pos']),
                                      self.convert_to_str(m['quat']), m['name'],
                                      m['name'], m['name'], m['material']) for i, m in enumerate(self.blocks)]
        body_str = '\n'.join(body_list)
        return body_str

    def convert_to_str(self, an_iterable):
        tmp = ""
        for an_item in an_iterable:
            tmp += str(an_item) + " "
        # tmp = " ".join(str(an_iterable))
        return tmp[:-1]

    def get_random_pos(self, height=None):
        x = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        y = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        if height is None:
            z = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
        else:
            z = height
        return np.array([x, y, z])

    def get_random_rbga(self, num_colors):
        rgb = list(pickRandomColor(num_colors))
        return rgb + [1]

    def initialize(self, use_cur_pos):
        tmp = MODEL_XML_BASE.format(self.get_asset_mesh_str(), self.get_asset_material_str(), self.get_body_str())
        model = load_model_from_xml(tmp)
        self.sim = MjSim(model)
        self._blank_observation = self.get_observation()
        if self.view:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

        # self.sim_state = self.sim.get_state()
        self._get_starting_step(use_cur_pos)

    def _get_starting_step(self, use_cur_pos):
        prev_positions = {}
        for i, aname in enumerate(self.names):
            if use_cur_pos:
                prev_positions[aname] = self.get_block_info(aname)["pos"]
            self.add_block(aname, [-5+i, -5+i, -5])

        for aname in self.names:
            if use_cur_pos:
                tmp_pos = prev_positions[aname]
                # print(aname, tmp_pos)
            else:
                tmp_pos = self.get_random_pos(self.drop_height)
            self.add_block(aname, tmp_pos)
            for i in range(self.internal_steps_per_step):
                self.internal_step()
                if self.view:
                    self.viewer.render()
            # self.sim_state = self.sim.get_state()

    ####Env internal step functions
    def add_block(self, ablock, pos):
        #pos (x,y,z)
        self.set_block_info(ablock, {"pos": pos})
        # self.sim.set_state(self.sim_state)

    def pick_block(self, pos):
        block_name = None
        for a_block in self.names:
            if self.intersect(a_block, pos):
                block_name = a_block

        if block_name is None:
            return False

        #PICK_LOC = np.array([0, 0, 5])
        #info = {"pos":PICK_LOC}
        #self.set_block_info(block_name, info)
        # self.sim.set_state(self.sim_state)
        return block_name

    def intersect(self, a_block, pos):
        cur_pos = self.get_block_info(a_block)["pos"]
        return np.max(np.abs(cur_pos - pos)) < self.TOLERANCE

    def get_block_info(self, a_block):
        info = {}
        info["poly"] = a_block[:-2]
        info["pos"] = np.copy(self.sim.data.get_body_xpos(a_block)) #np array
        info["quat"] = np.copy(self.sim.data.get_body_xquat(a_block))
        info["vel"] = np.copy(self.sim.data.get_body_xvelp(a_block))
        info["rot_vel"] = np.copy(self.sim.data.get_body_xvelr(a_block))
        return info

    def set_block_info(self, a_block, info):
        # print(a_block, info)
        # print("Setting state: {}, {}".format(a_block, info))
        sim_state = self.sim.get_state()
        start_ind = self.sim.model.get_joint_qpos_addr(a_block)[0]
        if "pos" in info:
            sim_state.qpos[start_ind:start_ind+3] = np.array(info["pos"])
        if "quat" in info:
           sim_state.qpos[start_ind+3:start_ind+7] = info["quat"]
        else:
            sim_state.qpos[start_ind + 3:start_ind + 7] = np.array([1, 0, 0, 0])

        start_ind = self.sim.model.get_joint_qvel_addr(a_block)[0]
        if "vel" in info:
            sim_state.qvel[start_ind:start_ind + 3] = info["vel"]
        else:
            sim_state.qvel[start_ind:start_ind + 3] = np.zeros(3)
        if "rot_vel" in info:
            sim_state.qvel[start_ind + 3:start_ind + 6] = info["rot_vel"]
        else:
            sim_state.qvel[start_ind + 3:start_ind + 6] = np.zeros(3)
        self.sim.set_state(sim_state)

    def internal_step(self, action=None):
        ablock = False
        if action is None:
            self.sim.forward()
            self.sim.step()
        else:
            pick_place = action[:3]
            drop_place = action[3:]

            ablock = self.pick_block(pick_place)
            if (ablock):
                # print("Dropping: {} {}".format(ablock, drop_place))
                self.add_block(ablock, drop_place)
        # self.sim_state = self.sim.get_state()
        return ablock


    ####Env external step functions
    # Input: action (4) or (6)
    # Output: resultant observation after taking the action
    def step(self, action):
        action = self._pre_process_actions(action)
        ablock = self.internal_step(action)
        # print(self.get_env_info())
        #if ablock:
        for i in range(self.internal_steps_per_step):
            self.sim.forward()
            self.sim.step()
            # self.internal_step()
            if self.view:
                self.viewer.render()

        # self.give_down_vel()
        # for i in range(200):
        #     self.sim.forward()
        #     self.sim.step()
        # self.sim_state = self.sim.get_state()

        # for aname in self.names: #This looks incorrect TODO: CHECK THIS
        #     self.add_block(aname, self.get_block_info(aname)["pos"])
        return self.get_observation()

    # Input: action can either be (A) or (T, A) where we want to execute T actions in a row
    # Output: Single obs
    def try_step(self, actions):
        tmp = self.get_env_info()
        # cur_state = copy.deepcopy(self.sim.get_state())
        if len(actions.shape) == 1:
            self.step(actions)
        elif len(actions.shape) == 2:
            for action in actions:
                self.step(action)
        else:
            raise KeyError("Wrong shape for actions: {}".format(actions.shape))
        obs = self.get_observation()
        # self.sim.set_state(cur_state)
        self.set_env_info(tmp)
        return obs

    def reset(self):
        self.names = []
        self.blocks = []
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            poly = np.random.choice(self.polygons)
            pos = self.get_random_pos()
            pos[2] = -2 * (i + 1)
            self.add_mesh(poly, pos, quat, self.get_random_rbga(self.num_colors))
        self.initialize(False)
        return self.get_observation()

    def get_observation(self):
        img = self.sim.render(self.img_dim, self.img_dim, camera_name="fixed")  # img is upside down, values btwn 0-255 (D,D,3)
        img = img[::-1, :, :]  # flips image right side up (D,D,3)
        return np.ascontiguousarray(img)  # values btwn 0-255 (D,D,3)

    def get_segmentation_masks(self):
        cur_obs = self.get_observation()
        tmp = np.abs(cur_obs - self._blank_observation) #(D,D,3)
        dif = np.where(tmp > 5, 1, 0).sum(2) #(D,D,3)->(D,D)
        dif = np.where(dif != 0, 1.0, 0.0)
        block_seg = dif
        background_seg = 1 - dif
        return np.array([background_seg, block_seg]) #Note: output btwn 0-1, (2,D,D)


    def get_obs_size(self):
        return [self.img_dim, self.img_dim]

    def get_actions_size(self):
        if self.include_z:
            return [6]
        else:
            return [4]

    # Inputs: actions (*,6)
    # Outputs: (*,6) if including z, (*,4) if not
    def _post_process_actions(self, actions):
        if self.include_z:
            return actions
        else:
            return actions[..., [0, 1, 3, 4]]

    # Inputs: actions (*,4), or (*,6)
    # Outputs: actions (*,6)
    def _pre_process_actions(self, actions):
        if actions.shape[-1] == 6:
            return actions

        full_actions = np.zeros(list(actions.shape)[:-1] + [6])  # (*,6)
        full_actions[..., [0, 1, 3, 4]] = actions
        full_actions[..., 2] = self.pick_height
        full_actions[..., 5] = self.drop_height
        return full_actions

    # Inputs: None
    # Outputs: Returns name of picked block
    #   If self.include z: Pick any random block
    #   Else: Picks a random block which can be picked up with the z pick set to self.pick_height
    def _get_rand_block_byz(self):
        if len(self.names) == 0:
            raise KeyError("No blocks in _get_rand_block_byz()!")
        if self.include_z:
            aname = np.random.choice(self.names)
        else:
            z_lim = self.pick_height
            tmp = [aname for aname in self.names if abs(self.get_block_info(aname)["pos"][2] - z_lim) < self.TOLERANCE]
            # while (len(tmp) == 0):
            #     z_lim += 0.5
            #     tmp = [aname for aname in self.names if self.get_block_info(aname)["pos"][2] <= z_lim]
            aname = np.random.choice(tmp)
            # print(tmp, aname)
        return aname

    # Input: action_type
    # Output: Single action either (6) or (4)
    def sample_action(self, action_type=None, include_noise=True):
        if len(self.names) == 1 and action_type == 'place_block':
            action_type = None

        if action_type == 'pick_block': #pick block, place randomly
            # aname = np.random.choice(self.names)
            aname = self._get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"]
            place = self.get_random_pos()
        elif action_type == 'place_block': #pick block, place on top of existing block
            # aname = np.random.choice(self.names)
            aname = self._get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"]  # + np.random.randn(3)/10
            names = copy.deepcopy(self.names)
            names.remove(aname)
            aname = np.random.choice(names)
            place = self.get_block_info(aname)["pos"]  # + np.random.randn(3)/10
            place[5] += 2  # Each block is roughly 1 unit wide
        elif action_type == 'remove_block':
            aname = self._get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"]  # + np.random.randn(3)/50
            place = [0, 0, -5]  # Place the block under the ground to remove it from scene
        elif action_type is None:
            if self.include_z:
                pick = self.get_random_pos()
                place = self.get_random_pos()
            else:
                pick = self.get_random_pos(self.pick_height)
                place = self.get_random_pos(self.drop_height)
        else:
            raise KeyError("Wrong input action_type!")
        ac = np.array(list(pick) + list(place))
        if include_noise:
            ac += np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=6) * 1.4
        # pdb.set_trace()
        # ac[2] = 0.6
        # ac[5] = self.drop_height
        return self._post_process_actions(ac)

    #Input: mean (*), std (*)
    #Output: actions (*)
    def sample_action_gaussian(self, mean, std):
        random_a = np.random.normal(mean, std)
        # # set pick height
        # random_a[2] = 0.6
        # # set place heights
        # random_a[5] = self.drop_height
        return random_a

    # # Input: mean (*), std (*)
    # # Output: actions (*)
    # def sample_multiple_action_gaussian(self, mean, std, num_samples):
    #     #mean and std should be (T, A)
    #     random_a = np.random.normal(mean, std, [num_samples] + list(mean.shape))
    #     # set pick height
    #     random_a[:, :, 2] = 0.6
    #     # set place height
    #     random_a[:, :, 5] = self.drop_height
    #     return random_a

    def move_blocks_side(self):
        # Move blocks to either side
        z = self.drop_height
        side_pos = [
            [-2.2, 1.5, z],
            [2.2, 1.5, z],
            [-2.2, 3.5, z],
            [2.2, 3.5, z]]
        # self.bounds = {'x_min':-2.5, 'x_max':2.5, 'y_min': 1.0, 'y_max' :4.0, 'z_min':0.05, 'z_max'2.2}
        place_lst = []
        for i, block in enumerate(self.names):
            place = copy.deepcopy(self.get_block_info(block)["pos"])
            place[-1] = self.drop_height
            self.add_block(block, side_pos[i])
            place_lst.append(place)
            #true_actions.append(side_pos[i] + list(place)) #Note pick & places z's might be
            # slightly
            #  off
        # sort by place height so place lowest block first

        for i in range(self.internal_steps_per_step):
            self.internal_step()
            if self.view:
                self.viewer.render()
        true_actions = []
        for i, block in enumerate(self.names):
            pick = self.get_block_info(block)["pos"]
            pick[-1] = 0.6
            place = place_lst[i]
            true_actions.append(np.concatenate([pick, place]))


        sorted(true_actions, key=lambda x : x[5])
        # print(true_actions)

        return self._post_process_actions(true_actions)


    def create_tower_shape(self):

        def get_valid_width_pos(width):
            num_pos = len(self.heights)
            possible = []
            for i in range(num_pos):
                valid = True
                for k in range(max(i - width, 0), min(i + width + 1, num_pos)):
                    if self.types[k] == "tetrahedron":
                        valid = False
                        break
                    if self.heights[i] < self.heights[k]:
                        valid = False
                        break
                    if self.heights[i] >= 3:
                        valid = False
                        break
                if valid:
                    possible.append(i)
            return possible

        def get_drop_pos(index):
            delta_x = 1
            y_val = 3
            left_most_x = -2.5
            return [left_most_x + index * delta_x, y_val, 4]

        self.names = []
        self.blocks = []

        self.heights = [0, 0, 0, 0, 0]
        self.types = [None] * 5
        self.check_clear_width = {'cube' : 1, 'horizontal_rectangle' : 1, 'tetrahedron' : 1}
        self.add_height_width = {'cube' : 0, 'horizontal_rectangle' : 1, 'tetrahedron' : 0}

        tmp_polygons = copy.deepcopy(self.polygons) #['cube', 'horizontal_rectangle', 'tetrahedron'][:2]

        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            poly = np.random.choice(tmp_polygons)
            tmp = get_valid_width_pos(self.check_clear_width[poly])
            if len(tmp) == 0:
                tmp_polygons.remove(poly)
                if len(tmp_polygons) == 0:
                    # print("DONE!")
                    break
                else:
                    continue

            tmp_polygons = copy.deepcopy(self.polygons)
            ind = np.random.choice(tmp)
            # print(poly, tmp, ind)
            self.update_tower_info(ind, poly)
            tmp_pos = get_drop_pos(ind)
            self.add_mesh(poly, tmp_pos, quat, self.get_random_rbga(self.num_colors))
        self.num_objects = len(self.names)
        self.initialize(True)

    def update_tower_info(self, ind, poly):
        self.types[ind] = poly
        width = self.add_height_width[poly]
        new_height = self.heights[ind] + 1
        for i in range(max(ind-width,0), min(ind+width+1, len(self.heights))):
            # print(i, new_height)
            self.heights[i] = new_height

        for i in range(1,4):
            if self.heights[i-1] == self.heights[i+1] and new_height == self.heights[i-1]:
                self.heights[i] = self.heights[i-1]

        # print(poly, ind, self.types, self.heights)

    def get_env_info(self):
        env_info = {}
        env_info["names"] = copy.deepcopy(self.names)
        env_info["blocks"] = copy.deepcopy(self.blocks)
        for i, aname in enumerate(self.names):
            info = self.get_block_info(aname)
            env_info["blocks"][i]["pos"] = copy.deepcopy(info["pos"])
            env_info["blocks"][i]["quat"] = copy.deepcopy(info["quat"])
        return env_info

    def set_env_info(self, env_info):
        self.names = env_info["names"]
        self.blocks = env_info["blocks"]
        self.initialize(True)

    def compute_accuracy(self, true_data, threshold=0.2):
        mjc_data = copy.deepcopy(true_data)

        max_err = -float('inf')
        data = self.get_env_info()

        for pred_datum in data['blocks']:
            err, mjc_match, err_pos, err_rgb = self._best_obj_match(pred_datum, mjc_data['blocks'])
            # del mjc_data[mjc_match]

            # print(err)
            if err > max_err:
                max_err = err
                max_pos = err_pos
                max_rgb = err_rgb

            if len(mjc_data) == 0:
                break

        correct = max_err < threshold
        return correct

    def _best_obj_match(self, pred, targs):
        def np_mse(x1, x2):
            return np.square(x1 - x2).mean()

        pos = pred['pos']
        rgb = pred['rgba']

        best_err = float('inf')
        for obj_data in targs:
            obj_name = obj_data['name']
            obj_pos = obj_data['pos']
            obj_rgb = obj_data['rgba']

            pos_err = np_mse(pos, obj_pos)
            rgb_err = np_mse(np.array(rgb), np.array(obj_rgb))
            err = pos_err + rgb_err

            if err < best_err:
                best_err = err
                best_obj = obj_name
                best_pos = pos_err
                best_rgb = rgb_err

        return best_err, best_obj, best_pos, best_rgb

    # #Assume only 1 block
    # def get_left_right_left(self):
    #     left_loc = np.array([self.wall_pos[0]-1.5, self.wall_pos[1]+1, self.pick_height])
    #     right_loc = np.array([self.wall_pos[0]+1.5, self.wall_pos[1]+1, self.pick_height])
    #
    #     cur_loc = self.get_block_info('cube_0')["pos"]




def createSingleSim(args):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    num_blocks = np.random.randint(args.min_num_objects, args.max_num_objects+1)
    myenv = BlockPickAndPlaceEnv(num_blocks, args.num_colors, args.img_dim, args.include_z, random_initialize=True, view=False)
    myenv.img_dim = args.img_dim
    # global myenv
    imgs = []
    acs = []
    initial_env_info = myenv.get_env_info()
    imgs.append(myenv.get_observation())
    for t in range(args.num_frames-1):
        if args.remove_objects == 'True':
            ac = myenv.sample_action('remove_block')
        else:
            rand_float = np.random.uniform()
            if rand_float < args.force_pick:
                ac = myenv.sample_action('pick_block')
            elif rand_float < args.force_pick + args.force_place:
                ac = myenv.sample_action('place_block')
            else:
                ac = myenv.sample_action(None)
        imgs.append(myenv.step(ac))
        acs.append(ac)

    acs.append(myenv.sample_action(None))

    values = {
        'features': np.array(imgs), #(T,D,D,3)
        'actions': np.array(acs),
        'env': initial_env_info
    }
    return values
    # return np.array(imgs), np.array(acs)

def create_left_right_left(env):
    def combine(pick_loc, place_loc):
        return np.concatenate((pick_loc, place_loc))

    left_loc = np.array([-1.5, env.wall_pos[1] + 2.5, env.pick_height])
    right_loc = np.array([1.5, env.wall_pos[1] + 2.5, env.pick_height])
    cur_loc = env.get_block_info('cube_0')["pos"]

    obs = [] #(T=3,D,D,3)
    segs = [] #(T=3,2,D,D)
    obs.append(env.step(combine(cur_loc, left_loc))) #Move from current position to left
    segs.append(env.get_segmentation_masks())

    obs.append(env.step(combine(left_loc, right_loc))) #Move from left position to right (behind wall)
    segs.append(env.get_segmentation_masks())

    obs.append(env.step(combine(right_loc, left_loc))) #Move from right to left
    segs.append(env.get_segmentation_masks())

    obs = np.expand_dims(np.array(obs), 1) #(T=3,D,D,3) -> (T,1,D,D,3)
    segs = np.tile(np.expand_dims(np.array(segs), 4), (1, 1, 1, 1, 3)) #(T,2,D,D)->(T,2,D,D,3)
    actions = np.array([combine(left_loc, right_loc), combine(right_loc, left_loc), combine(right_loc, left_loc)]) #(T=3,6)
    actions = actions[:, [0, 1, 3, 4]] #(T=2,4)
    return obs, segs, actions


def create_occlusion_dataset(args):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    env = BlockPickAndPlaceEnv(1, args.num_colors, args.img_dim, include_z=False, random_initialize=True, view=False)
    initial_env_info = env.get_env_info()
    obs, segs, actions = create_left_right_left(env) #(T,1,D,D,3), (T,2,D,D,3), (T,A=4), T=3 for all

    values = {
        'features': obs.squeeze(1), #(T,D,D,3)
        'actions': np.array(actions), #(T,A)
        'segs': segs,
        'env': initial_env_info
    }
    return values


"""
python rlkit/envs/blocks/mujoco/block_pick_and_place.py -f data/pickplace50k.h5 -nmin 3 -nmax 4 -nf 2 -ns 50000 -fpick 0.3 -fplace 0.4
"""
#########Start Unit Tests##########
def test_try_step():
    env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False, random_initialize=True)
    num_actions = 2
    rand_actions = np.array([env.sample_action("pick_block") for _ in range(num_actions)])

    ncols = num_actions+1
    nrows = 3
    cur_fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 6))

    axes[0, 0].imshow(env.get_observation() / 255, interpolation='nearest')
    axes[1, 0].imshow(env.get_observation() / 255, interpolation='nearest')
    axes[2, 0].imshow(env.get_observation() / 255, interpolation='nearest')

    # ###Testing try_step()
    for i in range(num_actions):
        # print(rand_actions[:i+1])
        axes[0, i + 1].set_title("Try step")
        # print(env.get_env_info())
        axes[0, i + 1].imshow(env.try_step(rand_actions[:i+1]) / 255, interpolation='nearest')
        # print(env.get_env_info())

        axes[2, i + 1].set_title("Env obs")
        axes[2, i + 1].imshow(env.get_observation() / 255, interpolation='nearest')
        # axes[0, i+1].set_title("{:2f}".format(rand_actions[i][0]))

    # print("HELLO")
    for i in range(num_actions):
        # print(rand_actions[i])
        # print(env.sim.get_state())
        # print(env.get_env_info())
        axes[1, i+1].imshow(env.step(rand_actions[i]) / 255, interpolation='nearest')
        axes[1, i+1].set_title("Env step")
        # print(env.get_env_info())
        # print(env.sim.get_state())
    cur_fig.savefig("test_try_step")

def test_env_infos():
    env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False, random_initialize=True)
    obs = []
    env_infos = []

    num_to_save = 4
    for i in range(num_to_save):
        # pdb.set_trace()
        obs.append(env.get_observation())
        env_infos.append(env.get_env_info())
        env.step(env.sample_action('pick_block', include_noise=False))

    new_env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False, random_initialize=False)
    new_obs = []
    for i in range(num_to_save):
        new_env.set_env_info(env_infos[i])
        new_obs.append(new_env.get_observation())

    plot_numpy(np.array([obs, new_obs]), file_name="test_env_infos")
    print(np.sum(np.power(np.array(obs) - np.array(new_obs), 2)))
    #Note that the pixelwise error is NOT zero even though images look visually the same
    #The loaded environment is nearly identical but not completely

def test_occlusion():
    env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64, include_z=False, random_initialize=True, view=False)
    background, block = env.get_segmentation_masks() #Each (D,D)
    background = np.tile(np.expand_dims(background, 2), (1,1,3)) #(D,D)->(D,D,1)->(D,D,3)
    block = np.tile(np.expand_dims(block, 2), (1, 1, 3))  # (D,D)->(D,D,1)->(D,D,3)
    # pdb.set_trace()
    plot_numpy(np.array([[background, block], [env.get_observation(), env._blank_observation]]), "test_occlusion.png")

def test_occlusion_dataset():
    env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64, include_z=False, random_initialize=True, view=False)
    obs, segs, actions = create_left_right_left(env) #(T=3,1,D,D,3), (T=3,2,D,D,3)
    tmp = np.concatenate((obs, segs), axis=1) #(T,3,D,D,3)
    plot_numpy(tmp, "test_occlusion_dataset.png")


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
#########End Unit Tests##########
def create_entire_dataset():
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, required=True)
    parser.add_argument('-i', '--img_dim', type=int, default=64)
    parser.add_argument('-ns', '--num_sims', type=int, default=2)
    parser.add_argument('-mi', '--make_images', type=bool, default=False)
    parser.add_argument('-c', '--num_colors', type=int, default=None)
    parser.add_argument('--output_path', default='', type=str, help='path to save images')
    parser.add_argument('-p', '--num_workers', type=int, default=1)
    args = parser.parse_args()

    if args.filename[-3:] == ".h5":
        args.filename = args.filename[:-3]

    args.num_frames = 3

    print(args)
    # single_sim_func = lambda : createSingleSim(args)
    single_sim_func = create_occlusion_dataset
    # createSingleSim(args)
    env = BlockPickAndPlaceEnv(1, 1, args.img_dim, include_z=False, random_initialize=True)
    ac_size = env.get_actions_size()
    obs_size = env.get_obs_size()

    # myenv = BlockPickAndPlaceEnv(2, args.num_colors, args.img_dim, args.include_z,
    #                              random_initialize=True, view=False)
    dgu.createMultipleSims(args, obs_size, ac_size, single_sim_func, num_workers=int(args.num_workers))

    dgu.hdf5_to_image(args.filename + '.h5')
    for i in range(min(10, args.num_sims)):
        tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
        dgu.make_gif(tmp, "animation.gif")


# python block_pick_and_place.py -f pickplace_multienv_10k -nmin 2 -nmax 2 -nf 21 -ns 10000 -fpick 0.3 -fplace 0.4
# python block_pick_and_place.py -f pickplace_multienv_c3_10k -nmin 2 -nmax 2 -nf 21 -ns 10000 -fpick 0.3 -fplace 0.4 -c 3 -p 1

# python block_pick_and_place.py -f pickplace_1and2_1k -nmin 1 -nmax 2 -nf 21 -ns 1000 -fpick 0.4 -fplace 0.3

if __name__ == '__main__':
    # test_occlusion_dataset()
    # test_try_step()
    # test_env_infos()
    create_entire_dataset()



    ###Testing loading###
    # with open('local_test.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(1 * 6, 1 * 6))
    # b = BlockPickAndPlaceEnv(6, None, 64, include_z=False, random_initialize=False, view=False)
    # for i in range(0, 2):
    #     tmp = data['training'][i]
    #     b.set_env_info(tmp)
    #     print(tmp)
    #     axes[i].imshow(b.get_observation(), interpolation='nearest')
    #     imfile = os.path.join("local_test_recon_{}.png".format(i))
    #     cv2.imwrite(imfile, b.get_observation())
    #     # plot_multi_image(np.array([[b.get_observation()]]), 'testing.png')
    # cur_fig.savefig("testing")

    # num_rows = 2
    # cur_fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(num_rows * 6, 1 * 6))
    # b = BlockPickAndPlaceEnv(3, None, 64, include_z=False, random_initialize=True, view=False)
    # for i in range(20):
    #     ac = b.sample_action("pick_block")
    #     print(ac[2])
    #     b.step(ac)
    # for i in range(5):
    #     obs = []
    #     # if i == 0:
    #     #     b.create_tower_shape()
    #     #     obs.append(b.get_observation())
    #     #     b.move_blocks_side()
    #     # else:
    #     #     b.create_tower_shape()
    #     #     b.move_blocks_side()
    #         # b.step(b.sample_action("pick_block"))
    #     b.create_tower_shape()
    #     obs.append(b.get_observation())
    #     b.move_blocks_side()
    #     obs.append(b.get_observation())
    #     for k in range(num_rows):
    #         axes[k, i].imshow(obs[k], interpolation='nearest')
    # cur_fig.savefig("HELLO")

    # num_rows = 1
    # cur_fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(num_rows * 6, 1 * 6))
    #
    # b = BlockPickAndPlaceEnv(4, None, 64, False, random_initialize=True, view=False)
    # b.create_tower_shape()
    # ob = b.get_observation()
    # axes[0].imshow(ob, interpolation='nearest')
    #
    # env_info = b.get_env_info()
    # # print(env_info)
    #
    # c = BlockPickAndPlaceEnv(2, None, 64, False, random_initialize=False, view=False)
    # c.set_env_info(env_info)
    # # print(c.get_env_info())
    # axes[1].imshow(c.get_observation(), interpolation='nearest')
    # cur_fig.savefig("HELLO")

    # b=c
    # for i in range(10):
    #     # pdb.set_trace()
    #     # b.step()
    #     # if i == 2000:
    #     #     cur_pos = b.get_block_info("cube_0")["pos"]
    #     #     ac = list(cur_pos) + [-0.5, -0.5, 1]
    #     #     b.step(ac)
    #     if i < 2000:# and i %100 == 0:
    #         # cur_pos = b.get_block_info("cube_0")["pos"]
    #         # ac = list(cur_pos) + [-0.5, -0.5, 1]
    #         b.step(b.sample_action("pick_block"))
    #         # print(i)
    #         # for aname in b.names:
    #         #     print(b.get_block_info(aname))
    #     # b.viewer.render()



    #Running and rendering example
    # b = BlockPickAndPlaceEnv(4, None, 64, True, True, view=True)
    # # b = BlockPickAndPlaceEnv(4, None, 64, include_z, random_initialize=False, view=False)
    # for i in range(10):
    #     # pdb.set_trace()
    #     # b.step()
    #     # if i == 2000:
    #     #     cur_pos = b.get_block_info("cube_0")["pos"]
    #     #     ac = list(cur_pos) + [-0.5, -0.5, 1]
    #     #     b.step(ac)
    #     if i < 2000:# and i %100 == 0:
    #         # cur_pos = b.get_block_info("cube_0")["pos"]
    #         # ac = list(cur_pos) + [-0.5, -0.5, 1]
    #         b.step(b.sample_action("pick_block"))
    #         # print(i)
    #         # for aname in b.names:
    #         #     print(b.get_block_info(aname))
    #     b.viewer.render()
    #
    # #Plotting images
    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))
    #
    # myenv = BlockPickAndPlaceEnv(4, None, 64, include_z=False)
    # myenv.create_tower_shape()
    # obs = []
    # obs.append(myenv.get_observation())
    # axes[0].imshow(obs[-1], interpolation='nearest')
    # cur_fig.savefig("HELLO")
    # # plt.show()
    #
    # # the_input = input("Suffix: ")
    #
    # for i in range(4):
    #     # an_action = myenv.sample_action()
    #     # tmp_name = myenv.step(an_action)
    #     # while (not tmp_name):
    #     #     an_action = myenv.sample_action()
    #     #     tmp_name = myenv.step(an_action)
    #     ac = myenv.sample_action("pick_block")
    #     tmp = myenv.step(ac)
    #     print(tmp.shape, np.max(tmp), np.min(tmp), np.mean(tmp))
    #     # print(np.max(tmp), np.min(tmp), np.mean(tmp))
    #     axes[i+1].set_title(ac)
    #     axes[i + 1].imshow(tmp)#, interpolation='nearest')
    #     # for k in range(5):
    #     #     axes[i+1, k].imshow(tmp[k], interpolation='nearest')
    # cur_fig.savefig("HELLO")
