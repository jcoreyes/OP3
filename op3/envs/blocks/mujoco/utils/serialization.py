import os
import pickle
import numpy as np
import scipy.misc

from torch.autograd import Variable

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def scale_images(images, size):
	scaled = []
	for ind in range(images.shape[0]):
		img = scipy.misc.imresize(images[ind], size=(size,size), interp='nearest')
		scaled.append(img)
	scaled = np.stack(scaled)
	return scaled

def read_image(path, img_dim = None):
    img = scipy.misc.imread(path)
    if img_dim is not None:
    	img = scipy.misc.imresize(img, size=(img_dim,img_dim), interp = 'nearest')
    if img.shape[-1] == 4:
        img = img[:,:,:-1]
    img = img.transpose(2,0,1) / 255.
    img = torch.Tensor(img)
    return img

def save_image(path, img):
	if type(img) == Variable:
		img = img.data
	if type(img) != np.ndarray:
		img = img.cpu().numpy()
		img = img.transpose(1,2,0)
		img = img * 255.
	img = img.astype(np.uint8)
	scipy.misc.imsave(path, img)

def read_pickle(path, encoding = None):
    f = open(path, 'rb')
    if encoding is not None:
    	param = pickle.load(f, encoding = encoding)
    else:
    	param = pickle.load(f)
    return param