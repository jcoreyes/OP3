import os
import pdb
import numpy as np
import shutil
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import op3.envs.blocks.mujoco.utils.data_generation_utils as dgu
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer
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
        {}
    </worldbody>
</mujoco>
"""

# COLOR_LIST discrete colors: Red, Lime, Blue, Yellow, Cyan, Magenta, Black, White
COLOR_LIST = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [1, 1, 1], [255, 255, 255]]
def pickRandomColor(an_int):
    if an_int is None:
        return np.random.uniform(low=0.0, high=1.0, size=3)
    tmp = np.random.randint(0, an_int)
    return np.array(COLOR_LIST[tmp])/255


class BlockPickAndPlaceEnv:
    def __init__(self, num_objects, num_colors, img_dim, include_z, random_initialize=False, view=False):
        self.asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mujoco_data/stl/')
        self.img_dim = img_dim
        self.include_z = include_z
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron'][:1]
        self.num_colors = num_colors
        self.num_objects = num_objects
        self.view = view

        # Hyper-parameters
        self.internal_steps_per_step = 2000
        self.drop_height = 5
        self.pick_height = 0.59
        self.bounds = {'x_min': -2.5, 'x_max': 2.5, 'y_min': 1.0, 'y_max': 4.0, 'z_min': 0.05, 'z_max': 2.2}
        self.TOLERANCE = 0.2

        self.names = []
        self.blocks = []

        if random_initialize:
            self.reset()

    #### Env initialization functions
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
        if self.view:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

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
            else:
                tmp_pos = self.get_random_pos(self.drop_height)
            self.add_block(aname, tmp_pos)
            for i in range(self.internal_steps_per_step):
                self.internal_step()
                if self.view:
                    self.viewer.render()

    #### Env internal step functions
    def add_block(self, ablock, pos):
        #pos (x,y,z)
        self.set_block_info(ablock, {"pos": pos})

    def pick_block(self, pos):
        block_name = None
        for a_block in self.names:
            if self.intersect(a_block, pos):
                block_name = a_block

        if block_name is None:
            return False
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


    #### Env external step functions
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

    # Input: actions (B,A)
    # Output: Largest manhattan distance component to closest block (B)
    def get_action_error(self, actions):
        vals = np.ones(actions.shape[0])*10000
        for i, an_action in enumerate(actions):
            for a_block in self.names:
                vals[i] = min(np.max(np.abs(self.get_block_info(a_block)["pos"][:2] - an_action[:2])), vals[i])
        return vals

    # Resets the environment
    def reset(self):
        self.names = []
        self.blocks = []
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            poly = np.random.choice(self.polygons)
            pos = self.get_random_pos()
            pos[-2] += -2 * (i + 1)
            self.add_mesh(poly, pos, quat, self.get_random_rbga(self.num_colors))
        self.initialize(False)
        return self.get_observation()

    # Returns an observation
    def get_observation(self):
        img = self.sim.render(self.img_dim, self.img_dim, camera_name="fixed")  # img is upside down, values btwn 0-255 (D,D,3)
        img = img[::-1, :, :]  # flips image right side up (D,D,3)
        return np.ascontiguousarray(img)  # values btwn 0-255 (D,D,3)

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
            return np.array(actions)[..., [0, 1, 3, 4]]

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
            aname = np.random.choice(tmp)
        return aname

    # Input: action_type
    # Output: Single action either (6) or (4)
    def sample_action(self, action_type=None, pick_noise_ratio=0.0, place_noise_ratio=0.0):
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
            pick = self.get_block_info(aname)["pos"] #+ np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=6) * 0.5
            names = copy.deepcopy(self.names)
            names.remove(aname)
            aname = np.random.choice(names)
            place = self.get_block_info(aname)["pos"] #+ np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=6)
            place[2] += 3  # Each block is roughly 1 unit wide
        elif action_type == 'remove_block':
            aname = self._get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"]  # + np.random.randn(3)/50
            place = [0, 0, -5]  # Place the block under the ground to remove it from scene
        # elif action_type == "noisy_pick":
        #     aname = self._get_rand_block_byz()
        #     pick = self.get_block_info(aname)["pos"] #+ np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=6) * 0.5
        #     place = self.get_random_pos()
        # elif action_type == "noisy_miss":
        #     aname = self._get_rand_block_byz()
        #     pick = self.get_block_info(aname)["pos"] #+ np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=6) * 5
        #     place = self.get_random_pos()
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

        if pick_noise_ratio:
            ac[:3] += np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=3) * pick_noise_ratio
        else:
            ac[3:] += np.random.uniform(-self.TOLERANCE, self.TOLERANCE, size=3) * place_noise_ratio
        return self._post_process_actions(ac)


    # Input: mean (*), std (*), num_actions
    # Output: actions (num_actions, *)
    def sample_multiple_action_gaussian(self, mean, std, num_actions):
        # return np.stack([self.sample_action_gaussian(mean, std) for _ in range(num_actions)])
        ans = np.random.normal(mean, std, size=[num_actions] + list(mean.shape))

        ## Clip actions to stay in bounds
        if not self.include_z:
            ans[..., 0] = np.clip(ans[..., 0], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 1] = np.clip(ans[..., 1], self.bounds['y_min'], self.bounds['y_max'])
            ans[..., 2] = np.clip(ans[..., 2], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 3] = np.clip(ans[..., 3], self.bounds['y_min'], self.bounds['y_max'])
        else:
            ans[..., 0] = np.clip(ans[..., 0], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 1] = np.clip(ans[..., 1], self.bounds['y_min'], self.bounds['y_max'])
            ans[..., 2] = np.clip(ans[..., 2], self.bounds['z_min'], self.bounds['z_max'])
            ans[..., 3] = np.clip(ans[..., 3], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 4] = np.clip(ans[..., 4], self.bounds['y_min'], self.bounds['y_max'])
            ans[..., 5] = np.clip(ans[..., 5], self.bounds['z_min'], self.bounds['z_max'])
        return ans

    # Input: mean (*, 2/3), std (*, 2/3), num_actions
    # Output: actions (num_actions, *, 2/3)
    def sample_multiple_place_locs_gaussian(self, mean, std, num_actions):
        ans = np.random.normal(mean, std, size=[num_actions] + list(mean.shape))

        ## Clip actions to stay in bounds
        if not self.include_z:
            ans[..., 0] = np.clip(ans[..., 0], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 1] = np.clip(ans[..., 1], self.bounds['y_min'], self.bounds['y_max'])
        else:
            ans[..., 0] = np.clip(ans[..., 0], self.bounds['x_min'], self.bounds['x_max'])
            ans[..., 1] = np.clip(ans[..., 1], self.bounds['y_min'], self.bounds['y_max'])
            ans[..., 2] = np.clip(ans[..., 2], self.bounds['z_min'], self.bounds['z_max'])
        return ans


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
        colors = []
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
            while True:
                color = self.get_random_rbga(self.num_colors)
                # if len(colors) > 0:
                #    import pdb; pdb.set_trace()
                if len(colors) == 0 or np.linalg.norm(color[:3] - np.array(colors)[:, :3]).min(0) > 0.3:
                    break
            colors.append(color)
            self.add_mesh(poly, tmp_pos, quat, color)
        self.num_objects = len(self.names)
        self.initialize(True)

    def update_tower_info(self, ind, poly):
        self.types[ind] = poly
        width = self.add_height_width[poly]
        new_height = self.heights[ind] + 1
        for i in range(max(ind-width, 0), min(ind+width+1, len(self.heights))):
            self.heights[i] = new_height

        for i in range(1,4):
            if self.heights[i-1] == self.heights[i+1] and new_height == self.heights[i-1]:
                self.heights[i] = self.heights[i-1]


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

    # Output: If check_all_in_bounds is false, return actions(N,3)
    #  Else return true if all the boxes are in bounds, false otherwise
    def get_block_locs(self, check_all_in_bounds=False):
        ans = []
        for a_block in self.names:
            ans.append(self.get_block_info(a_block)["pos"])  # (3)
        ans = np.array(ans)  # (Num_blocks,3)
        if check_all_in_bounds:
            x_max = np.max(ans[:, 0])
            x_min = np.min(ans[:, 0])
            y_max = np.max(ans[:, 1])
            y_min = np.min(ans[:, 1])
            if x_max > self.bounds['x_max'] or x_min < self.bounds['x_min'] or y_max > self.bounds['y_max'] or \
                    y_min < self.bounds['y_min']:
                return False
            else:
                return True
        return ans

    # Input: Computes accuracy (#blocks correct/total #) of the current environment given the true data and threshold
    def compute_accuracy(self, true_data, threshold=0.2):
        mjc_data = copy.deepcopy(true_data)

        max_err = -float('inf')
        data = self.get_env_info()

        correct = 0
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

            if err < threshold:
                correct += 1

        correct /= float(len(data['blocks']))
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



def createSingleSim(args):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    num_blocks = np.random.randint(args.min_num_objects, args.max_num_objects+1)
    myenv = BlockPickAndPlaceEnv(num_blocks, args.num_colors, args.img_dim, args.include_z, random_initialize=True, view=False)
    myenv.img_dim = args.img_dim
    noise_ratio = args.noise

    imgs = []
    acs = []
    initial_env_info = myenv.get_env_info()
    imgs.append(myenv.get_observation())
    for t in range(args.num_frames-1):
        if np.random.uniform() < noise_ratio:
            pick_noise_ratio = 5
        else:
            pick_noise_ratio = 0.5

        if args.remove_objects == 'True':
            ac = myenv.sample_action('remove_block', pick_noise_ratio=pick_noise_ratio)
        else:
            rand_float = np.random.uniform()
            if rand_float < args.force_pick:
                ac = myenv.sample_action('pick_block', pick_noise_ratio=pick_noise_ratio)
            elif rand_float < args.force_pick + args.force_place:
                ac = myenv.sample_action('place_block', pick_noise_ratio=pick_noise_ratio, place_noise_ratio=1)
            else:
                ac = myenv.sample_action(None, pick_noise_ratio=pick_noise_ratio)
        imgs.append(myenv.step(ac))
        acs.append(ac)

    acs.append(myenv.sample_action(None))

    values = {
        'features': np.array(imgs),
        'actions': np.array(acs),
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

    ## Testing try_step()
    for i in range(num_actions):
        axes[0, i + 1].set_title("Try step")
        axes[0, i + 1].imshow(env.try_step(rand_actions[:i+1]) / 255, interpolation='nearest')

        axes[2, i + 1].set_title("Env obs")
        axes[2, i + 1].imshow(env.get_observation() / 255, interpolation='nearest')

    for i in range(num_actions):
        axes[1, i+1].imshow(env.step(rand_actions[i]) / 255, interpolation='nearest')
        axes[1, i+1].set_title("Env step")
    cur_fig.savefig("test_try_step")

def test_env_infos():
    env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False, random_initialize=True)
    obs = []
    env_infos = []

    num_to_save = 4
    for i in range(num_to_save):
        obs.append(env.get_observation())
        env_infos.append(env.get_env_info())
        env.step(env.sample_action('pick_block'))

    new_env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False, random_initialize=False)
    new_obs = []
    for i in range(num_to_save):
        new_env.set_env_info(env_infos[i])
        new_obs.append(new_env.get_observation())

    plot_numpy(np.array([obs, new_obs]), file_name="test_env_infos")
    print(np.sum(np.power(np.array(obs) - np.array(new_obs), 2)))
    # Note that the pixelwise error is NOT zero even though images look visually the same
    # The loaded environment is nearly identical but not completely


# Inputs: numpy_array (H,W,D,D,3)
def plot_numpy(numpy_array, file_name, titles=None):
    num_rows, num_cols = numpy_array.shape[:2]
    cur_fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 6))
    for y in range(num_rows):
        for x in range(num_cols):
            axes[y, x].imshow(numpy_array[y, x], interpolation='nearest')
            if titles is not None:
                axes[y, x].set_title(titles[y, x])
    cur_fig.savefig(file_name)
#########End Unit Tests##########

def create_entire_dataset():
    # Examples:
    # python block_pick_and_place.py -f pickplace_multienv_10k -nmin 2 -nmax 2 -nf 21 -ns 10000 -fpick 0.3 -fplace 0.4
    # python block_pick_and_place.py -f pickplace_multienv_c3_10k -nmin 2 -nmax 2 -nf 21 -ns 10000 -fpick 0.3 -fplace 0.4 -c 3
    # python block_pick_and_place.py -f pickplace_1and2_1k -nmin 1 -nmax 2 -nf 21 -ns 1000 -fpick 0.4 -fplace 0.3
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, required=True)  # Filename to save .h5 dataset
    parser.add_argument('-nmin', '--min_num_objects', type=int, default=1)    # Min blocks in a scene
    parser.add_argument('-nmax', '--max_num_objects', type=int, default=2)    # Max blocks in a scene
    parser.add_argument('-i', '--img_dim', type=int, default=64)              # Square image observations
    parser.add_argument('-nf', '--num_frames', type=int, default=21)          # Number of frames per simulation
    parser.add_argument('-ns', '--num_sims', type=int, default=2)             # Number of simulations to run
    parser.add_argument('-mi', '--make_images', type=bool, default=False)     # Whether to visualize the entire dataset or not
    parser.add_argument('-c', '--num_colors', type=int, default=None)         # None means we pick continuous colors. A number specifies discrete colors (see COLOR_LIST)
    parser.add_argument('-fpick', '--force_pick', type=float, default=0.5)    # Action distribution hyper-parameter
    parser.add_argument('-fplace', '--force_place', type=float, default=0.5)  # Action distribution hyper-parameter
    parser.add_argument('-noise', '--noise', type=float, default=0)           # Action distribution hyper-parameter
    parser.add_argument('-r', '--remove_objects', type=bool, default=False)   # *@ A different type of action that removes blocks
    parser.add_argument('-z', '--include_z', type=bool, default=False)        # *@ Whether or not to include the z axis in the action space
    parser.add_argument('--output_path', default='', type=str, help='path to save images')  # Path for saving images
    parser.add_argument('-p', '--num_workers', type=int, default=1)           # Number of parallel workers when creating the dataset. Note that this may actually be slower with p>1, so I recommend keeping p=1.
    # Note: *@ signifies that we did not include experiments with these in the paper but include this for possible modification and extra experimentation
    args = parser.parse_args()

    if args.filename[-3:] == ".h5":
        args.filename = args.filename[:-3]

    print(args)
    info = {}
    info["min_num_objects"] = args.min_num_objects
    info["max_num_objects"] = args.max_num_objects
    info["img_dim"] = args.img_dim

    if args.remove_objects:  # See *@ note
        args.num_frames = 2
    info["num_frames"] = args.num_frames

    env = BlockPickAndPlaceEnv(1, 1, args.img_dim, args.include_z, random_initialize=True)
    ac_size = env.get_actions_size()
    obs_size = env.get_obs_size()

    dgu.createMultipleSims(args, obs_size, ac_size, createSingleSim, num_workers=int(args.num_workers))

    num_to_visualize = min(10, args.num_sims)
    dgu.hdf5_to_image(args.filename + '.h5', num_to_visualize=num_to_visualize)
    for i in range(num_to_visualize):
        tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
        dgu.make_gif(tmp, "animation.gif")

if __name__ == '__main__':
    # test_try_step()
    # test_env_infos()
    create_entire_dataset()
