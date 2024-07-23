# -*- coding: utf-8 -*-
# @Time    : 2023/10/30 12:43
# @Author  : Zhenjie Tang
# @File    : inversion_kd_head.py
# @Description: 添加注释
import os
import h5py
import torch
import time
import pickle as pk
import numpy as np
from scipy.stats import chi
from model_definition import Decoder
from GroundwaterModelPytorch.Modflow_FDM3_class import ModflowFDM3
from GroundwaterModelPytorch.MT3DMS_FDM3_class import Mt3dmsFDM3


def gen_welspd(torch_device):
    # create N samples
    N = 1000

    # generate wells releasing the contaminant
    y_wel = np.array([5, 15, 25, 35,
                      5, 15, 25, 35,
                      5, 15, 25, 35])
    x_wel = np.array([5, 5, 5, 5,
                      11, 11, 11, 11,
                      17, 17, 17, 17])
    wells = {i: [y_wel[i], x_wel[i]] for i in range(len(y_wel))}

    # sample N wells
    np.random.seed(888)
    indexes = np.random.choice(12, N)
    locations = [wells[indexes[i]] for i in range(N)]

    values = np.random.uniform(low=100, high=1000, size=(N, 5)).astype(int)

    location = locations[0]
    value = values[0]

    welspd = {}
    for i in range(5):
        strength_variable = torch.tensor(value[i], device=torch_device, dtype=torch.float64)
        welspd[i] = [(3, location[0], location[1], strength_variable, -1)]

    return welspd


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


def init_wells(n_strenth, torch_device):
    # Set the all possible source position values
    y_wel = np.array([5, 15, 25, 35,
                      5, 15, 25, 35,
                      5, 15, 25, 35])
    x_wel = np.array([5, 5, 5, 5,
                      11, 11, 11, 11,
                      17, 17, 17, 17])
    wells = {i: [y_wel[i], x_wel[i]] for i in range(len(y_wel))}

    # initializing the strength inversion variable
    strength_variables = []
    for i in range(n_strenth):
        strength_variable = torch.tensor(0.5, device=torch_device, dtype=torch.float64, requires_grad=True)
        strength_variables.append(strength_variable)

    stress_period_data_variables = {}
    for i in range(n_strenth):
        temp = []
        for j in range(len(wells)):
            temp.append((3, wells[j][0], wells[j][1], strength_variables[i], -1))

        stress_period_data_variables[i] = temp

    return stress_period_data_variables, strength_variables, len(wells)


def add_noise(obs_data, device):
    shape = obs_data.size()
    np.random.seed(888)
    obs_data_noise = obs_data + 0.05 * torch.std(obs_data) * torch.tensor(np.random.normal(0, 1, shape), dtype=torch.float64, device=device)

    return obs_data_noise


# Hyperparameter settings
torch_device = torch.device('cpu')
if torch.cuda.is_available():
    torch_device = torch.device('cuda')
    print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torch_device))
n_iter = 80    # number of iterations
lr = 0.025
continue_train = False
nf, d, h, w = 2, 2, 11, 21   # latent Dimensions
latentZ_D = chi(nf * d * h * w).mean()  # 2, 2, 11, 21

"""
Generate ground truth
"""
modeflow_model, mt3dms_model = MT3DMS_model(torch_device)

# create head_gt from modeflow_model
with open('../dataset/test_K_data/test_kds.pkl', 'rb') as file:
    kds = np.asarray(pk.load(file))[:2200]
#np.random.seed(888)
idx = 1946 #np.random.choice(2200)
kd = kds[idx]   # [6, 41, 81]
# kd[kd < 0] = 0   # 跟原始论文一样
kd = np.exp(kd)
kd_gt = torch.as_tensor(kd, dtype=torch.float,  device=torch_device)
output_gt = modeflow_model.run(kd_gt)
head_gt = output_gt.Phi
Q_gt = output_gt.Q
Qx_gt = output_gt.Qx
Qy_gt = output_gt.Qy
Qz_gt = output_gt.Qz

# create conc_gt from mt3dms_model
welspd = gen_welspd(torch_device)
position_gt = [int(welspd[0][0][1]), int(welspd[0][0][2])]    # [y, x]
strength_gt = [welspd[i][0][3] for i in range(len(welspd))]
conc_gt = mt3dms_model.run(output_gt, welspd)
conc_gt[conc_gt < 0] = 0  # pay attention

# Get the value of the observation well
y_obs = np.linspace(1, 39, num=20).astype('int')
x_obs = np.linspace(1, 79, num=40).astype('int')
head_gt_obs = add_noise(head_gt[:, y_obs, :][:, :, x_obs], device=torch_device)
conc_gt_obs = add_noise(conc_gt[:, :, y_obs, :][:, :, :, x_obs], device=torch_device)

# save result
result_dir = "./inversion_step1/idx_" + str(idx)
os.makedirs(result_dir, exist_ok=True)

"""
iterative inversion
"""
# loss functions
relu = torch.nn.ReLU()
pixel_loss = torch.nn.MSELoss()

# initialize latent variable and the decoder network
checkpoint_path = result_dir + "/checkpoint26.pt"
if continue_train:
    checkpoint_data = torch.load(checkpoint_path)
    latent_z = checkpoint_data["latent_z"]
    position_weight = checkpoint_data["position"]
    strength_variables = checkpoint_data["strength"]
    stress_period_data_variables = checkpoint_data["spd"]
    n_positions = checkpoint_data["n_positions"]

    print("Restore checkpoint from" + checkpoint_path)

else:
    latent_z = torch.tensor(np.random.normal(0, 1, (1, nf, d, h, w)), dtype=torch.float, requires_grad=True, device=torch_device)  # pay attention to the dtype
    stress_period_data_variables, strength_variables, n_positions = init_wells(len(welspd), torch_device)
    position_weight = torch.zeros(n_positions, dtype=torch.float64, requires_grad=True, device=torch_device)

decoder = Decoder(inchannels=nf)
decoder.load_state_dict(torch.load('./exp_2024063011/N23000_Bts64_Eps50_lr0.0002_lw0.01/AAE_decoder_epoch50.pth'))
decoder.to(torch_device)
decoder.eval()


# Optimizers
optimizer = torch.optim.Adam(strength_variables + [position_weight, latent_z], lr=lr)
torch.autograd.set_detect_anomaly(True)

loss_result = []
for i in range(n_iter):
    start = time.time()

    position_probability = torch.softmax(position_weight, dim=0)

    # Forward calculation
    kd_log = torch.squeeze(decoder(latent_z))
    kd_pre = torch.exp(kd_log)
    output_pre = modeflow_model.run(kd_pre)
    head_pre = output_pre.Phi[:, y_obs, :][:, :, x_obs]
    Q_pre = output_pre.Q
    Qx_pre = output_pre.Qx
    Qy_pre = output_pre.Qy
    Qz_pre = output_pre.Qz

    conc_pre = mt3dms_model.run_parallel(output_pre, stress_period_data_variables, n_positions)
    conc_pre[conc_pre < 0] = 0  # [10, n_positions, 6, 41, 81]
    conc_pre = torch.reshape(conc_pre, (10, n_positions, 6 * 41 * 81))  # [10, n_positions, 6*41*81]
    conc_pre = torch.transpose(conc_pre, 1, 2)  # [10, 6*41*81, n_positions]
    conc_pre = torch.matmul(conc_pre, position_probability)  # [10, 6*41*81]
    conc_pre = torch.reshape(conc_pre, (10, 6, 41, 81))  # [10, 6, 41, 81]
    conc_pre = conc_pre[:, :, y_obs, :][:, :, :, x_obs]

    # loss function
    gaussian_mean_loss = relu(torch.norm(latent_z) - latentZ_D) ** 2
    head_loss = pixel_loss(head_pre, head_gt_obs)
    conc_loss = pixel_loss(conc_pre, conc_gt_obs)
    strength_stack = torch.stack(strength_variables, dim=0)
    expert_loss = torch.mean(relu(torch.abs(strength_stack) - 1) ** 2)

    # position regularization loss
    # entropy_loss = -torch.sum(position_probability * torch.log2(position_probability)) * 0.1

    loss = head_loss + conc_loss + gaussian_mean_loss + expert_loss

    loss_result.append(loss.detach().cpu().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i + 1)
    print('time for training:', time.time() - start)
    print("total_loss: %f, head_loss %f, conc_loss %f, gaussian_mean_loss %f, expert_loss %f"
          % (loss, head_loss, conc_loss, gaussian_mean_loss, expert_loss))
    print(position_probability.detach().cpu().numpy())

    # save the result
    if (i + 1) % 1 == 0:

        check_point_data = {"latent_z": latent_z, "position": position_weight, "strength": strength_variables,
                            "spd": stress_period_data_variables, "n_positions": n_positions}
        torch.save(check_point_data, result_dir + "/checkpoint" + str(i + 1) + ".pt")

        f_out = h5py.File(result_dir + "/inversion_step1_result" + str(i + 1) + ".hdf5", "w")
        f_out.create_dataset('latent_z', data=latent_z.detach().cpu().numpy(), dtype='f', compression='gzip')
        f_out.create_dataset('kd_gt', data=kd_gt.detach().cpu().numpy(), dtype='f', compression='gzip')
        f_out.create_dataset('kd_pre', data=kd_pre.detach().cpu().numpy(), dtype='f', compression='gzip')
        f_out.create_dataset('probability', data=position_probability.detach().cpu().numpy(), dtype='f',
                             compression='gzip')
        strength_stack = torch.stack(strength_variables, dim=0)
        f_out.create_dataset('strength_variables', data=strength_stack.detach().cpu().numpy(), dtype='f',
                             compression='gzip')
        f_out.close()

        error = np.mean(np.abs(kd_gt.detach().cpu().numpy() - kd_pre.detach().cpu().numpy()))

        print("###################################################################")
        print("average error:", error)
        print("###################################################################")

        f_out = h5py.File(result_dir + "/loss_result.hdf5", "w")
        f_out.create_dataset('loss_result', data=loss_result, dtype='f', compression='gzip')
        f_out.close()

