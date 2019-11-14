import math
import random
import numpy as np

def quat_to_axangle(quat):
    r, i, j, k = quat
    l2 = math.sqrt( i**2 + j**2 + k**2 )

    if l2 == 0:
        ## angle is zero
        ## axis can be any normalized vector
        a_x = 1
        a_y = 0
        a_z = 0

        theta = 0
    else:
        a_x = i / l2
        a_y = j / l2
        a_z = k / l2

        theta = 2 * math.atan2(l2, r)

    axangle = [a_x, a_y, a_z, theta]
    return axangle

def axangle_to_quat(axangle):
    a_x, a_y, a_z, theta = axangle

    half_theta = theta / 2.
    r = math.cos(half_theta)
    i = math.sin(half_theta) * a_x
    j = math.sin(half_theta) * a_y
    k = math.sin(half_theta) * a_z

    quat = [r, i, j, k]

    return quat

def random_axangle(axis = None):
    theta = random.uniform(0, math.pi)
    if axis is None:
        axis = normalize( uniform( [0, 1], [0, 1], [0, 1]) )
    else:
        axis = normalize(axis)
    axangle = axis + [theta]
    return axangle

def random_quaternion(theta = None, axis = None):
    if theta is None:
        theta = random.uniform(0, math.pi)
    if axis is None:
        axis = normalize( uniform( [0, 1], [0, 1], [0, 1]) )

    a_x, a_y, a_z = axis
    axangle = [a_x, a_y, a_z, theta]

    quat = axangle_to_quat(axangle)
    return quat

def random_euler():
    pi = math.pi
    euler = uniform( [0, 2*pi], [0, 2*pi], [0, 2*pi] )
    return euler

def state_to_params(state):
    assert state.size(2) == 17
    
    params = {}
    params['translate'] = state[:, :, :3]
    params['rotate'] = state[:, :, 3:7]
    params['scale'] = state[:, :, 7]
    params['texture'] = state[:, :, 8:11]
    params['velp'] = state[:, :, 11:14]
    params['velr'] = state[:, :, 14:17]

    for k, v in params.items():
        params[k] = v.contiguous()
    return params

def params_to_state_single(params):
    # try: 
    state = torch.cat(  (   params['translate'], 
                            params['rotate'], 
                            params['scale'], 
                            params['texture'], 
                            params['velp'], 
                            params['velr']
                        ),  dim = -1
                    )

    state = state.contiguous()
    return state

def params_to_state(params):
    if type(params) == dict:
        state = params_to_state_single(params)
    elif type(params) == list:
        states_list = [params_to_state_single(p) for p in params]
        states = torch.cat(states_list, dim = -1)
    return states

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mv(src, dest):
    while os.path.exists(dest):
        folder = dest.split('/')[-1]
        folder += '_'
        dest = os.path.join(dest, '..', folder)
    shutil.move(src, dest)

def uniform(*bounds):
    samples = []
    for bound in bounds:
        l, h = bound
        sample = np.random.uniform(l, h)
        samples.append(sample)
    if len(samples) == 1: samples = samples[0]
    return samples

def normalize(vector):
    magnitude = math.sqrt(sum([v**2 for v in vector]))
    return [v / magnitude for v in vector]

def tensor_norm(tensor, dim = 1):
    norm = torch.sqrt( torch.sum( torch.pow(tensor, 2), dim = dim, keepdim = True) )
    return norm

def normalize_axangle(axangle):
    axis = axangle[:,:3]
    norm = tensor_norm(axis, dim = 1)
    axis = axis / norm
    out = torch.cat( (axis, axangle[:,-1].unsqueeze(1)), dim = 1 )

    norm_check = tensor_norm(axis)
    assert (norm_check.gt(.99) * norm_check.lt(1.01)).all()

    return out

def sample_rgba(*hsv_bounds):
    hsv = uniform(*hsv_bounds)
    rgba = list(colorsys.hsv_to_rgb(*hsv)) + [1]
    return rgba



    