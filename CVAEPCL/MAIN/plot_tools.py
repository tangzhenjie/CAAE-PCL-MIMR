import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import cm
import matplotlib.colors


def simple_plot(c_map, title=''):
    nx = 81
    ny = 41
    Lx = 2500
    Ly = 1250

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    if len(c_map) == 41:
        fig, axs = plt.subplots(1,1)
        # axs.set_xlabel('x(m)')
        # axs.set_ylabel('y(m)')
        # axs.set_xlim(0,Lx)
        # axs.set_ylim(0,Ly)
        c01map = axs.imshow(c_map, cmap='jet',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  vmin=c_map.min(), vmax=c_map.max(),
                  origin='lower')
        fig.colorbar(c01map, ax=axs, shrink=0.62)
    else:
        fig, axs = plt.subplots(len(c_map)//3, 3, figsize=(7, 2.5))
        axs = axs.flat
        for i, ax in enumerate(axs):
            # ax.set_xlim(0, Lx)
            # ax.set_ylim(0, Ly)
            c01map = ax.imshow(c_map[i], cmap='jet', interpolation='nearest',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      vmin=c_map[i].min(), vmax=c_map[i].max(),
                      origin='lower')
            ax.set_axis_off()
            v1 = np.linspace(np.min(c_map[i]), np.max(c_map[i]), 5, endpoint=True)
            fig.colorbar(c01map, ax=ax, fraction=0.021, pad=0.04, ticks=v1)

    plt.suptitle(title)
    name = title + '.pdf'
    plt.tight_layout()
    plt.show()
    fig.savefig(name, format='pdf', bbox_inches='tight')

    return

def plot_pred(samples, epoch, idx, output_dir):
    Ncol = 3
    Nrow = samples.shape[0] // Ncol

    fig, axes = plt.subplots(Nrow, Ncol, figsize=(Ncol*4, Nrow*2.1))
    fs = 16  # font size
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if j < samples.shape[0]:

            cax = ax.imshow(samples[j], cmap='jet', origin='lower', vmin=np.min(samples), vmax=np.max(samples))
            cbar = plt.colorbar(cax, ax=ax, fraction=0.025, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
            cbar.ax.tick_params(axis='both', which='both', length=0)
            cbar.ax.yaxis.get_offset_text().set_fontsize(fs-3)
            cbar.ax.tick_params(labelsize=fs-2)

    plt.savefig(output_dir+'/epoch_{}_{}.png'.format(epoch, idx), bbox_inches='tight',dpi=600)
    plt.close(fig)

    print("epoch {}, done printing".format(epoch))

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

if __name__ == '__main__':
    import pickle as pk
    import torch

    def load_data(n_train, n_test, batch_size):
        with open('../dataset/train_K_data/kds.pkl', 'rb') as file:
            kds = np.expand_dims(np.asarray(pk.load(file)), axis=1)
        print('Total number of conductivity images:', len(kds))

        x_train = kds[:n_train]
        x_test = kds[n_train: n_train + n_test]
        print("total training data shape: {}".format(x_train.shape))

        data = torch.utils.data.TensorDataset(torch.FloatTensor(x_train))
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                  shuffle=True, num_workers=int(2))
        return data_loader, x_test

    _, x_test = load_data(20, 20, 20)
    c_map = x_test[np.random.randint(0, len(x_test), 1)[0]]
    simple_plot(c_map[0], 'log K')
    plot_pred(c_map[0], 0, 0, "./")
