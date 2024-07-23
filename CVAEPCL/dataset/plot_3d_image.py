# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 11:01
# @Author  : Zhenjie Tang
# @File    : plot_3d_image.py
# @Description:

import numpy as np

import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
import matplotlib.backends.backend_pdf

mpl.rcParams['figure.figsize'] = (8, 8)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
# plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
# Set the default serif font to Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'  # Ensure that the serif font is used
#plt.rcParams['text.usetex'] = True
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_3d(data, title='', cut=None):
    data = np.transpose(data, (2, 1, 0))
    data = np.flip(data, axis=2)
    filled = np.ones(data.shape)
    if cut is not None:
        filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0
    x, y, z = np.indices(np.array(filled.shape) + 1)

    v1 = np.linspace(np.min(data), np.max(data), 8, endpoint=True)
    norm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
    # ax.set_box_aspect([250, 125, 50])
    ax.set_box_aspect([180, 150, 120])

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04, ticks=v1, )
    ax.set_axis_off()
    plt.tight_layout()
    # ax.set_title(title)

    plt.show()
    input("Once you have adjusted the image, press Enter to save the high-resolution image.")

    # save image with high resolution
    fig.savefig('high_res_image.png', dpi=300, bbox_inches='tight')

    return fig



train_im = scipy.io.loadmat('./train_K/K4.mat')
train_im = train_im['K']  # image size 105 x 180 x 150

test_im = scipy.io.loadmat('./test_K/K4.mat')
test_im = test_im['K']  # image size 15 x 180 x 150


whole_img = np.vstack((train_im, test_im))
fig = plot_3d(whole_img, title='./logk_data', cut=None)
