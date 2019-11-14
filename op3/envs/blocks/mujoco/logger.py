import random
import numpy as np
import scipy.misc

from op3.envs.blocks.mujoco.XML import XML
import op3.envs.blocks.mujoco.contacts as contacts
import op3.envs.blocks.mujoco.utils as utils
from PIL import Image

class Logger:

    def __init__(self, xml, sim, steps = 100, img_dim = 64, albedo = False):
        self.sim = sim
        self.steps = steps
        self.img_dim = img_dim
        self.albedo_flag = albedo
        self.render_dim = self.img_dim * 2
        self.meshes = {}
        self.masks = {}
        for mesh in xml.meshes:
          mesh_name = mesh['name']
          mesh_log = {}
          mesh_log['xpos']  = np.zeros( (steps, 3) )
          mesh_log['xaxangle'] = np.zeros( (steps, 4) )
          mesh_log['xvelp'] = np.zeros( (steps, 3) )
          mesh_log['xvelr'] = np.zeros( (steps, 3) )

          mesh_log['xscale'] = np.zeros( (steps, 1) )
          mesh_log['xrgba'] = np.zeros( (steps, 3) )
          mesh_log['xscale'][:,:] = mesh['xscale']
          mesh_log['xrgba'][:,:] = mesh['xrgba'][:3]
          mesh_log['ply'] = mesh_name[:-2]
          self.meshes[mesh_name] = mesh_log
          self.masks[mesh_name] = np.zeros( (steps, self.img_dim, self.img_dim, 3) )

    def log_step(self, step):
        self.sim.forward()
        for mesh_name in self.meshes.keys():
         self.log_mesh(mesh_name, step)

    def log_mesh(self, mesh_name, step):
        xpos, xaxangle, xvelp, xvelr = self.get_body_data(mesh_name)

        self.meshes[mesh_name]['xpos'][step]  = xpos
        self.meshes[mesh_name]['xaxangle'][step] = xaxangle
        self.meshes[mesh_name]['xvelp'][step] = xvelp
        self.meshes[mesh_name]['xvelr'][step] = xvelr

    def get_body_data(self, mesh_name):
        ## position
        xpos  = self.sim.data.get_body_xpos(mesh_name)
        ## quaternion
        xquat = self.sim.data.get_body_xquat(mesh_name)
        ## positional velocity
        xvelp = self.sim.data.get_body_xvelp(mesh_name)
        ## rotational velocity
        xvelr = self.sim.data.get_body_xvelr(mesh_name)

        xaxangle = utils.quat_to_axangle(xquat)

        return xpos, xaxangle, xvelp, xvelr

    def log_image(self, step, camera = 'fixed'):
        image = self.sim.render(self.render_dim, self.render_dim, camera_name = camera)
        if self.img_dim != self.render_dim:
            image = np.array(Image.fromarray(image).resize((self.img_dim, self.img_dim))).astype(np.uint8)
        if 'images' not in dir(self):
            M, N, C = image.shape
            self.images = np.zeros( (self.steps, M, N, C) )
            self.albedo = np.zeros( (self.steps, M, N, C) )
        self.images[step] = image
        return image

    def log_albedo(self, step, camera = 'fixed'):
        rgba = self.sim.model.mat_rgba.copy()
        spec = self.sim.model.mat_specular.copy()
        emit = self.sim.model.mat_emission.copy()
        shin = self.sim.model.mat_shininess.copy()

        self.sim.model.mat_specular[:] = 0
        self.sim.model.mat_shininess[:] = 0

        mesh_names = self.masks.keys()
        for mesh in mesh_names:
            mesh_ind = self.sim.model._geom_name2id[mesh]
            mat_ind  = self.sim.model.geom_matid[mesh_ind]

            self.sim.model.mat_emission[mat_ind] = 1

        image = self.sim.render(self.render_dim, self.render_dim, camera_name = camera)
        if self.img_dim != self.render_dim:
            image = scipy.misc.imresize(image, size = (self.img_dim, self.img_dim))

        self.albedo[step] = image

        self.sim.model.mat_rgba[:] = rgba
        self.sim.model.mat_specular[:] = spec
        self.sim.model.mat_emission[:] = emit
        self.sim.model.mat_shininess[:] = shin

    def change_rgba(self, obj_name, rgba):
        mesh_ind = self.sim.model._geom_name2id[obj_name]
        mat_ind = self.sim.model.geom_matid[mesh_ind]

        old_rgba = self.sim.model.mat_rgba[mat_ind] = rgba

        return old_rgba

    def log_masks(self, step, camera = 'fixed'):
        rgba = self.sim.model.mat_rgba.copy()
        spec = self.sim.model.mat_specular.copy()
        emit = self.sim.model.mat_emission.copy()
        shin = self.sim.model.mat_shininess.copy()

        self.sim.model.mat_specular[:] = 0
        self.sim.model.mat_shininess[:] = 0

        mesh_names = self.masks.keys()
        for mesh in mesh_names:
            mesh_ind = self.sim.model._geom_name2id[mesh]
            mat_ind  = self.sim.model.geom_matid[mesh_ind]

            self.sim.model.mat_rgba[:] = [0, 0, 0, 1]
            self.sim.model.mat_rgba[mat_ind] = [1, 1, 1, 1]

            self.sim.model.mat_emission[mat_ind] = 1

            image = self.sim.render(self.render_dim, self.render_dim, camera_name = camera)
            if self.img_dim != self.render_dim:
                image = scipy.misc.imresize(image, size = (self.img_dim, self.img_dim))

            self.masks[mesh][step] = image > 0.5

        self.sim.model.mat_rgba[:] = rgba
        self.sim.model.mat_specular[:] = spec
        self.sim.model.mat_emission[:] = emit
        self.sim.model.mat_shininess[:] = shin

    def export_xml(self, xml):
        xml_new = XML()
        for mesh in xml.meshes:
            mesh_name = mesh['name']
            polygon = mesh['polygon']
            pos, axangle, _, _ = self.get_body_data(mesh_name)

            rgba = mesh['xrgba']
            scale = mesh['xscale']

            xml_new.add_mesh(polygon, scale = scale, pos = pos, axangle = axangle, rgba = rgba, name = mesh_name)
        return xml_new

    def log(self, step):
        self.log_step(step)
        self.log_image(step)
        #self.log_masks(step)
        if self.albedo_flag:
            self.log_albedo(step)

    def sample_object(self):
        mesh_names = list(self.masks.keys())
        return random.choice( mesh_names )

    def log_embedder(self, step, obj_name):
        transparent_names = [name for name in self.masks.keys() if name != obj_name]
        rgba_dict = {}
        for name in transparent_names:
            rgba = self.change_rgba(name, [0,0,0,0])
            rgba_dict[name] = rgba

        self.log_image(step)

        for name in transparent_names:
            rgba = rgba_dict[name]
            self.change_rgba(name, rgba)

        self.log_step(step)
        self.log_masks(step)

    def position_body(self, name, pos, axangle):
        joint_ind = self.sim.model._joint_name2id[name]
        quat = utils.axangle_to_quat(axangle)

        qpos_start_ind = joint_ind * 7
        qpos_end_ind   = (joint_ind+1) * 7

        qfrc_start_ind = joint_ind * 6
        qfrc_end_ind   = (joint_ind+1) * 6

        qpos = self.sim.data.qpos[qpos_start_ind:qpos_end_ind]
        qpos[:3] = pos
        qpos[3:] = quat

        self.sim.data.qfrc_constraint[qpos_start_ind:qpos_end_ind] = 0

        self.sim.forward()

    def step(self, steps):
        for i in range(steps):
            self.sim.step()

    def settle_sim(self, drop_name, min_steps, max_steps, vel_threshold = 0.1):
        step = 0
        for _ in range(min_steps):
            self.settle_step(drop_name)
            step += 1

        max_vel = np.abs(self.sim.data.qvel).max()
        while max_vel > vel_threshold:
            self.settle_step(drop_name)

            max_vel = np.abs(self.sim.data.qvel[:,]).max()
            step += 1
            if step > max_steps:
                break
        return step

    def settle_step(self, drop_name):
        joint_ind = self.sim.model._joint_name2id[drop_name]
        qpos_start_ind = joint_ind * 7
        qpos_end_ind   = (joint_ind+1) * 7
        qvel_start_ind = joint_ind * 6
        qvel_end_ind   = (joint_ind+1) * 6

        state = self.sim.data.qpos[qpos_start_ind:qpos_end_ind].copy()
        vel = self.sim.data.qvel[qvel_start_ind:qvel_end_ind]

        vel[:] = 0
        self.sim.forward()

        self.sim.step()
        self.sim.data.qpos[qpos_start_ind:qpos_end_ind] = state

    def remove_tower_overlaps(self, names):
        num_objects = len(names)
        for ind in range(1, num_objects):
            top = names[ind]
            bottom = names[ind-1]

            joint_ind = self.sim.model._joint_name2id[top]
            qpos_z_ind = (joint_ind * 7) + 2

            while contacts.are_overlapping(self.sim, bottom, top):
                self.sim.data.qpos[qpos_z_ind] += 0.01
                self.sim.forward()

    def check_stability(self, name, init_pos):
        joint_ind = self.sim.model._joint_name2id[name]
        qpos_start_ind = joint_ind * 7
        qpos_end_ind   = qpos_start_ind + 3

        qpos = self.sim.data.qpos[qpos_start_ind:qpos_end_ind]

        if np.max( np.abs(qpos - init_pos) ) < 0.5:
            return True
        else:
            return False



    def hold_drop_execute(self, hold_names, drop_name, steps, logger = None, start_log_step = 0):
        hold_dict = {}
        for hold_ind, hold_name in enumerate(hold_names):
            temp_pos_x = hold_ind * 10
            temp_pos_z = 10000
            joint_ind = self.sim.model._joint_name2id[hold_name]
            qpos_start_ind = joint_ind * 7
            qpos_end_ind   = (joint_ind+1) * 7
            qvel_start_ind = joint_ind * 6
            qvel_end_ind   = (joint_ind+1) * 6

            state = self.sim.data.qpos[qpos_start_ind:qpos_end_ind].copy()
            vel = self.sim.data.qvel[qvel_start_ind:qvel_end_ind]
            hold_dict[hold_name] = (qpos_start_ind, qpos_end_ind, qvel_start_ind, qvel_end_ind, state)

            ## set position away from rest of simulation
            ## and give 0 velocity
            self.sim.data.qpos[qpos_start_ind+1] = temp_pos_x
            self.sim.data.qpos[qpos_start_ind+2] = temp_pos_z
            vel[:] = 0

        while contacts.is_overlapping(self.sim, drop_name):
            joint_ind = self.sim.model._joint_name2id[drop_name]
            qpos_start_ind = joint_ind * 7

            self.sim.data.qpos[qpos_start_ind+2] += 0.05
            self.sim.forward()

        for i in range(steps):
            self.sim.forward()
            self.sim.step()
            if logger is not None:
                logger.log(start_log_step + i)

        for hold_name in hold_names:
            qpos_start_ind, qpos_end_ind, qvel_start_ind, qvel_end_ind, state = hold_dict[hold_name]
            self.sim.data.qpos[qpos_start_ind:qpos_end_ind] = state
            self.sim.data.qfrc_constraint[qvel_start_ind:qvel_end_ind] = 0
            self.sim.data.qvel[qvel_start_ind:qvel_end_ind] = 0
        self.sim.forward()

    def get_state(self):
        states = {}
        for ind, (name, mesh) in enumerate(self.meshes.items()):

            temp_pos_x = ind * 10
            temp_pos_z = 10000
            joint_ind = self.sim.model._joint_name2id[name]
            qpos_start_ind = joint_ind * 7
            qpos_end_ind   = (joint_ind+1) * 7

            state = self.sim.data.qpos[qpos_start_ind:qpos_end_ind].copy()

            states[name] = {'ply': mesh['ply'], 'qpos': state, 'scale': mesh['xscale'][0][0], 'rgba': mesh['xrgba'][0].tolist()}

        return states


    def hold_drop(self, hold_names, steps):
        hold_dict = {}
        for hold_ind, hold_name in enumerate(hold_names):
            temp_pos_x = hold_ind * 10
            temp_pos_z = 10000
            joint_ind = self.sim.model._joint_name2id[hold_name]
            qpos_start_ind = joint_ind * 7
            qpos_end_ind   = (joint_ind+1) * 7
            qvel_start_ind = joint_ind * 6
            qvel_end_ind   = (joint_ind+1) * 6

            state = self.sim.data.qpos[qpos_start_ind:qpos_end_ind].copy()
            vel = self.sim.data.qvel[qvel_start_ind:qvel_end_ind]
            hold_dict[hold_name] = (qpos_start_ind, qpos_end_ind, qvel_start_ind, qvel_end_ind, state)

            ## set position away from rest of simulation
            ## and give 0 velocity
            self.sim.data.qpos[qpos_start_ind+1] = temp_pos_x
            self.sim.data.qpos[qpos_start_ind+2] = temp_pos_z
            vel[:] = 0

        for i in range(steps):
            self.sim.forward()
            self.sim.step()

        for hold_name in hold_names:
            qpos_start_ind, qpos_end_ind, qvel_start_ind, qvel_end_ind, state = hold_dict[hold_name]
            self.sim.data.qpos[qpos_start_ind:qpos_end_ind] = state
            self.sim.data.qfrc_constraint[qvel_start_ind:qvel_end_ind] = 0
            self.sim.data.qvel[qvel_start_ind:qvel_end_ind] = 0
        self.sim.forward()


    def get_logs(self, step = None):
        if step is None:
            if self.albedo_flag:
                return self.meshes, self.images, self.masks, self.albedo
            else:
                return self.meshes, self.images, self.masks
        else:
            step_meshes = {}
            step_masks  = {}
            for mesh_name, mesh_log in self.meshes.items():
                step_log = {}
                for op, param in mesh_log.items():
                    step_log[op] = param[step][np.newaxis,:]
                step_meshes[mesh_name] = step_log
                step_masks[mesh_name] = self.masks[mesh_name][step]
            step_image = self.images[step]
            # return step_meshes, step_image, step_meshes #RV: Before, possible bug...
            return step_meshes, step_image, step_masks  # RV: After



    
