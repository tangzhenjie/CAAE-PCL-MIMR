import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from CVAEPCL.MAIN.model_definition import Decoder
from GroundwaterModelPytorch.Modflow_FDM3_class import ModflowFDM3
from GroundwaterModelPytorch.MT3DMS_FDM3_class import Mt3dmsFDM3
from tqdm import tqdm


def MT3DMS_model(torch_device):

    # specify a rectangular grid
    x = np.linspace(0, 2500, 82)
    y = np.linspace(0, 1250, 42)
    z = np.linspace(0, 300, 7)

    SHP = (len(z) - 1, len(y) - 1, len(x) - 1)


    """
    run the FDM3 modeflow model
    """
    FQ = np.zeros(SHP)  # all flows zero. Note sz is the shape of the model grid

    IH = np.zeros(SHP, dtype=np.float64)
    h_grad = 0.012
    l_head, r_head = h_grad * 2500, 0.
    IH[:, :, 0] = l_head
    IH[:, :, -1] = r_head

    IBOUND = np.ones(SHP)
    IBOUND[:, :, 0] = -1
    IBOUND[:, :, -1] = -1

    # the Steady state 3D Finite Difference Model
    ModflowClass = ModflowFDM3(x, y, z, FQ, IH, IBOUND, torch_device)

    """
    run the FDM3 mt3dms model
    """
    IC = np.zeros(SHP)
    ICBUND = np.ones(SHP)
    total_days = 40 * 365  # days
    transport_step_size = 4 * 365  # days can speed up
    save_step_size = 4 * 365
    stress_period_step_size = 4 * 365
    prsity = 0.3
    al = 35.  # meter
    trpt = 0.3
    trpv = 0.3
    Mt3dmsClass = Mt3dmsFDM3(x, y, z, total_days, ICBUND, IC, stress_period_step_size, transport_step_size,
                             save_step_size, torch_device, al=al, trpt=trpt, trpv=trpv, prsity=prsity)

    return ModflowClass, Mt3dmsClass

def sample_source(torch_device):
    # Set the all possible source position values
    y_wel = np.array([5, 15, 25, 35,
                      5, 15, 25, 35,
                      5, 15, 25, 35])
    x_wel = np.array([5, 5, 5, 5,
                      11, 11, 11, 11,
                      17, 17, 17, 17])
    wells = {i: [y_wel[i], x_wel[i]] for i in range(len(y_wel))}

    x_ind = np.random.choice(12)
    well = wells[x_ind]

    stress_period_data_variables = {}
    for i in range(5):
        strength_variable = torch.tensor(np.random.uniform(low=100, high=1000), device=torch_device, dtype=torch.float64)
        stress_period_data_variables[i] = [(3, well[0], well[1], strength_variable, -1)]

    return stress_period_data_variables

def plot_3d(data, cut=None):
    fig = plt.figure()
    if data.shape[0] == 6:
        vmin = np.min(np.array(data))
        vmax = np.max(np.array(data))
        v1 = np.linspace(vmin, vmax, 8, endpoint=True)

        data = np.transpose(data, (2, 1, 0))
        data = np.flip(data, axis=2)
        filled = np.ones(data.shape)
        if cut is not None:
            filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0

        x, y, z = np.indices(np.array(filled.shape) + 1)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data)), edgecolors=None)
        ax.set_box_aspect([250, 125, 50])
        ax.set_axis_off()
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        m.set_array([])

        fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04, ticks=v1)

    else:
        for i in range(1, len(data) + 1):
            data_new = np.transpose(data[i-1], (2, 1, 0))
            data_new = np.flip(data_new, axis=2)
            vmin = np.min(np.array(data_new))
            vmax = np.max(np.array(data_new))
            v1 = np.linspace(vmin, vmax, 8, endpoint=True)
            filled = np.ones(data_new.shape)
            if cut is not None:
                filled[cut[2]:, :cut[1], (6 - cut[0]):] = 0

            x, y, z = np.indices(np.array(filled.shape) + 1)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            ax = fig.add_subplot(2, 5, i, projection='3d')
            ax.voxels(x, y, z, filled, facecolors=plt.cm.jet(norm(data_new)), edgecolors=None)
            ax.set_box_aspect([250, 125, 50])
            ax.set_axis_off()
            m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
            m.set_array([])
            fig.colorbar(m, ax=ax, fraction=0.015, pad=0.04, ticks=v1)

    plt.tight_layout()
    plt.show()

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
    #fig.savefig(name, format='pdf', bbox_inches='tight')

    return

if __name__ == "__main__":
    torch_device = torch.device('cpu')
    if torch.cuda.is_available():
        torch_device = torch.device('cuda')
        print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
    print('Running on ' + str(torch_device))

    # initialize the models
    modeflow_model, mt3dms_model = MT3DMS_model(torch_device)
    nf, d, h, w = 2, 2, 11, 21   # latent Dimensions
    decoder = Decoder(inchannels=nf)
    decoder.load_state_dict(torch.load('../MAIN/exp_2024063011/N23000_Bts64_Eps50_lr0.0002_lw0.01/AAE_decoder_epoch50.pth'))
    decoder.to(torch_device)
    decoder.eval()

    N_simu = 4000
    n_index = 0
    Head_output = []
    Conc_output = []
    pbar = tqdm(total=N_simu)
    while n_index < N_simu:
        # sample the k from Decoder(z) z ~ N(0, 1) and position from 12 possible location (uniform distribution)
        # and the strength value from [100, 900] (uniform distribution)
        latent_z = torch.tensor(np.random.normal(0, 1, (1, nf, d, h, w)), dtype=torch.float, device=torch_device)
        kd_log = torch.squeeze(decoder(latent_z))
        kd = torch.exp(kd_log)
        stress_period_data_variables = sample_source(torch_device)

        output_modflow = modeflow_model.run(kd)
        head = output_modflow.Phi
        conc = mt3dms_model.run(output_modflow, stress_period_data_variables)
        conc[conc < 0] = 0

        Head_output.append(head.detach().cpu().numpy())
        Conc_output.append(conc.detach().cpu().numpy())

        n_index += 1
        pbar.update(1)

    pbar.close()
    ## save Head_output and Conc_output in hdf5 file
    hf = h5py.File('./MC_dataset/MC_dataset_{}.hdf5'.format(str(N_simu)), 'w')
    hf.create_dataset('head', data=Head_output, dtype='f', compression='gzip')
    hf.create_dataset('conc', data=Conc_output, dtype='f', compression='gzip')
    hf.close()
    print('\n', len(Head_output))








