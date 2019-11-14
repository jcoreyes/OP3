import numpy as np
import matplotlib.pyplot as plt



def plot_multi_image(imgs, save_dir, caption=None):
    # imgs is (h, w, imsize, imsize, 3), numpy arrays and not pytorch tensors!
    if imgs.shape[-1] != 3:
        imgs = np.moveaxis(imgs, 2, -1)
    if imgs.dtype is not np.uint8:
        imgs = np.clip((imgs * 255), 0, 255).astype(np.uint8)


    rows, cols, imsize, imsize, _ = imgs.shape

    fig = plt.figure(figsize=(9, 13))
    ax = []
    count = 1
    for i in range(rows):
        for j in range(cols):
            ax.append(fig.add_subplot(rows, cols, count))
            ax[-1].set_yticklabels([])
            ax[-1].set_xticklabels([])
            if caption is not None and caption[i][j] != 0:
                ax[-1].set_title('%0.4f' % caption[i][j])
            plt.imshow(imgs[i, j])
            count += 1

    plt.savefig(save_dir)
    plt.close('all')