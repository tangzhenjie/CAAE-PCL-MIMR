import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.colors
import os
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import cm

mpl.rcParams['figure.figsize'] = (8, 6)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
# plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
# Set the default serif font to Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'  # Ensure that the serif font is used
# plt.rcParams['axes.linewidth'] = 2.0 # 边框
#plt.rcParams['text.usetex'] = True
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_3d(data, vmin, vmax, title='', cut=None):
    # data preprocess
    data = np.transpose(data, (2, 1, 0))
    data = np.flip(data, axis=2)
    vmin = vmin    # np.min(np.array(data))
    vmax = vmax    # np.max(np.array(data))
    x, y, z = np.indices(np.array(data.shape) + 1)
    filled = np.ones(data.shape)
    if cut is not None:
        filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0

    # plots
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax1.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
    ax1.set_box_aspect([250, 125, 50])
    ax1.set_axis_off()

    # colorbar
    ticks = np.linspace(vmin, vmax, 7, endpoint=True)  # colorbar 坐标
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax1, fraction=0.015, pad=0.04, ticks=ticks)

    plt.tight_layout()
    plt.show()
    fig.savefig(title, dpi=300, bbox_inches='tight')

    return

def plot_histogram(data, title=''):
    data = np.reshape(data, (-1,))
    plt.figure(figsize=(7, 4))
    color_rgb = (0 / 255, 114 / 255, 189 / 255)
    plt.hist(data, bins=50, color=color_rgb)
    # disable y
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xlim(-3, 3)
    plt.xticks(fontsize=20)  # fontweight='bold'
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.show()

inversion_step2_dir = "./inversion_step2/idx_1946/"
save_dir = "./step2_plot_result/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if __name__ == '__main__':
    file_name = "inversion_step2_result1000.hdf5"
    f_out = h5py.File(inversion_step2_dir + file_name, "r")
    strength_pre = np.abs(np.array(f_out['strength_pre'])) * 900 + 100
    strength_gt = np.array(f_out['strength_gt'])
    kd_gt = np.array(f_out['kd_gt'])
    kd_pre = np.array(f_out['kd_pre'])
    latent_z = np.array(f_out['latent_z']).ravel()
    f_out.close()

    # The mean absolute relative error
    error_percent = np.mean(np.abs(strength_pre - strength_gt) / strength_gt) * 100

    print("Mean absolute relative error: %f" % error_percent)

    plot_histogram(latent_z, title=save_dir + "latent_z")
    print(np.mean(latent_z))
    print(np.var(latent_z))

    kd_all = [np.log(kd_gt), np.log(kd_pre)]
    vmin = np.min(np.array(kd_all))
    vmax = np.max(np.array(kd_all))
    plot_3d(np.log(kd_gt), vmin=vmin, vmax=vmax, title=save_dir + "logk_gt", cut=[3, 12 + 1, 20 - 1])
    plot_3d(np.log(kd_pre), vmin=vmin, vmax=vmax, title=save_dir + "logk_pre", cut=[3, 12 + 1, 20 - 1])


    categories = [r'$S_{s1}$', r'$S_{s2}$', r'$S_{s3}$', r'$S_{s4}$', r'$S_{s5}$']
    width = 0.35  # 柱子的宽度

    # 创建柱状图
    x = np.arange(len(categories))

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    color_gt = (128 / 255, 172 / 255, 249 / 255)  # RGB 128,172,249
    color_pre = (235 / 255, 145 / 255, 132 / 255)  # RGB 235,145,132

    axs.bar(x - width / 2, strength_gt, width, color=color_gt, label='Reference value', align='center', edgecolor='black', linewidth=0.5)  # alpha=0.6
    axs.bar(x + width / 2, strength_pre, width, color=color_pre, label='Reconstruction value', align='center', edgecolor='black', linewidth=0.5)  # alpha=0.6
    axs.set_xticks(x, categories)
    axs.set_yticks(np.linspace(0, 1000, 6))
    # axs.set_title('Comparison of Strength Inversion Results')
    # axs.set_xlabel('Strength')
    # axs.set_ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + "strengths", dpi=300, bbox_inches='tight')
    plt.show()

    """
                可视化损失函数
    """
    f_out = h5py.File(inversion_step2_dir + "loss_result.hdf5", "r")
    loss_result = np.log(np.array(f_out['loss_result']))
    f_out.close()

    Iterations = [i + 1 for i in range(len(loss_result[0:-1]))]
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.plot(Iterations, loss_result[0:-1], 'k', label='Loss values')
    # axs.set_title('Loss Values')
    axs.set_xlabel('Iterations')
    # axs.set_ylabel('Loss')
    axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    axs.grid(True)  # 添加网格线
    plt.xlim(-20, 1000)  # 设置 x 轴范围从 0 到 80
    # axs.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'ln_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
