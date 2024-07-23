import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.colors
import pickle as pk
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# setup the setting for plotting, for example front size and so on
mpl.rcParams['figure.figsize'] = (8, 4)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
#plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
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
def plot_3d(real, gen, title='', cut=None):
    fig = plt.figure()
    data_all = [real, gen]
    vmin = np.min(np.array(data_all))
    vmax = np.max(np.array(data_all))

    v1 = np.linspace(vmin, vmax, 8, endpoint=True)
    i = 1
    for data in data_all:
        data = np.transpose(data, (2, 1, 0))
        data = np.flip(data, axis=2)
        filled = np.ones(data.shape)
        if cut is not None:
            filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0
        x, y, z = np.indices(np.array(filled.shape) + 1)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
        ax.set_box_aspect([250, 125, 50])
        ax.set_axis_off()
        i += 1
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04, ticks=v1, )

    plt.tight_layout()
    # ax.set_title(title)
    plt.savefig(title + '.pdf', bbox_inches='tight')
    return fig

def well_plot(c_map, axs, obs1, s_loc):
    nx = 82
    ny = 42
    Lx = 2500
    Ly = 1250


    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # axs.set_xlabel('x[m]')
    # axs.set_ylabel('y[m]')
    axs.set_xlim(0, Lx)
    axs.set_ylim(0, Ly)


    axs.imshow(c_map, cmap='jet', extent=[x.min(), x.max(), y.min(), y.max()],
              vmin=c_map.min(), vmax=c_map.max(), origin='lower')

    axs.scatter(obs1["x"], obs1["y"], c='yellow', edgecolors='black', label='possible source locations')

    axs.scatter(s_loc["x"], s_loc["y"], c='black', marker='D', label='reference source location')


    axs.legend(bbox_to_anchor=(1.02, 1.0), loc='lower right', ncol=3, frameon=False, columnspacing=1.5)

    return
if __name__ == '__main__':
    # create head_gt from modeflow_model
    with open('../dataset/test_K_data/test_kds.pkl', 'rb') as file:
        kds = np.asarray(pk.load(file))[:2200]
    idx = 1946
    ln_k = kds[idx]  # [6, 41, 81]
    k = np.exp(ln_k)
    cut = [3, 12 + 1, 20 - 1]
    k_fourth_layer = k[3]

    # preprocess the data
    k = np.transpose(k, (2, 1, 0))
    k = np.flip(k, axis=2)
    vmin = np.min(np.array(k))
    vmax = np.max(np.array(k))
    x, y, z = np.indices(np.array(k.shape) + 1)
    filled = np.ones(k.shape)
    if cut is not None:
        filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0

    # plots
    fig = plt.figure(figsize=(10, 8))

    ###################################################### 3d plot ##############################################
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax1.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(k)), edgecolors=None)
    ax1.set_box_aspect([250, 125, 50])
    ax1.set_axis_off()

    ###################################################### plot the well ##############################################
    y_wel_points = np.linspace(5, 35, num=4).astype('int')
    x_wel_points = np.linspace(5, 17, num=3).astype('int')
    y_wel_points, x_wel_points = np.meshgrid(y_wel_points, x_wel_points)

    dx = 2500 / 81
    dy = 1250 / 41
    obs = {"x": list(dx / 2 + x_wel_points * dx), "y": list(dy / 2 + y_wel_points * dy)}

    # the reference contaminant source location
    source_location_y = np.asarray(25)
    source_location_x = np.asarray(17)
    s_loc = {"x": dx / 2 + source_location_x * dx, "y": dy / 2 + source_location_y * dy}

    ax2 = fig.add_subplot(1, 2, 2)
    well_plot(k_fourth_layer, ax2, obs, s_loc)

    #################################################### final step for plotting ####################################
    ticks = np.linspace(vmin, vmax, 8, endpoint=True)  # colarbar 坐标
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.3)
    fig.colorbar(m, cax=cax, ax=ax2, fraction=0.015, pad=0.04, ticks=ticks)

    plt.subplots_adjust(wspace=0.001)
    plt.tight_layout()
    plt.show()
    fig.savefig('k_source.png', dpi=300, bbox_inches='tight')









    #plot_3d(kd_gt, kd_inver, title=inver_kd_result_dir + 'iter_' + str(100), cut=[3, 12 + 1, 20 - 1])




