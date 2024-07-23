# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 22:30
# @Author  : Zhenjie Tang
# @File    : DataVisualization.py
# @Description: the tools for data visualization

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


mpl.rcParams['figure.figsize'] = (8, 8)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
#plt.rcParams['text.usetex'] = True
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

def simple_plot(c_map, title=''):
    nx = 81
    ny = 41
    Lx = 2500
    Ly = 1250

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    if len(c_map) == 41:
        fig, axs = plt.subplots(1, 1)
        # axs.set_xlabel('x(m)')
        # axs.set_ylabel('y(m)')
        # axs.set_xlim(0,Lx)
        # axs.set_ylim(0,Ly)
        c01map = axs.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax=c_map.max(), origin='lower')
        fig.colorbar(c01map, ax=axs, shrink=0.62)
    else:
        fig, axs = plt.subplots(len(c_map)//3, 3, figsize=(7, 2.5))
        axs = axs.flat
        for i, ax in enumerate(axs):
            # ax.set_xlim(0, Lx) x轴视图限制
            # ax.set_ylim(0, Ly) y轴视图限制
            c01map = ax.imshow(c_map[i], cmap='jet', interpolation='nearest',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      vmin=c_map[i].min(), vmax=c_map[i].max(), origin='lower')
            ax.set_axis_off()
            v1 = np.linspace(np.min(c_map[i]),np.max(c_map[i]), 5, endpoint=True)
            fig.colorbar(c01map, ax=ax, fraction=0.021, pad=0.04, ticks=v1)

    plt.suptitle(title)
    #name = title + '.pdf'
    plt.tight_layout()
    #fig.savefig('images/'+name, format='pdf',bbox_inches='tight')
    plt.show()
    return fig


def plot_3d(data, title='', cut=None):
    data = np.transpose(data, (2, 1, 0))
    data = np.flip(data, axis=2)
    filled = np.ones(data.shape)
    if cut is not None:
        filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0
    x, y, z = np.indices(np.array(filled.shape) + 1)

    v1 = np.linspace(np.min(data), np.max(data), 8, endpoint=True)
    norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
    # ax.set_box_aspect([250, 125, 50])
    ax.set_box_aspect([150, 180, 105])

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04, ticks=v1, )
    ax.set_axis_off()
    plt.tight_layout()
    # ax.set_title(title)
    # plt.savefig(title+'.pdf',bbox_inches='tight')
    plt.show()
    return fig