# -*- coding: utf-8 -*-
# @Time    : 2023/9/12 11:26
# @Author  : Zhenjie Tang
# @File    : Modflow_FDM3.py
# @Description: Computing flows and heads of steady-state flow using the finite difference model, like the MODFLOW
import numpy as np
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
import torch
from torch_sparse_solve import solve
import sparse_linear_systems.sparse_solver as tzj_solve


class ModflowFDM3(object):
    def __init__(self, x, y, z, FQ, IH, IBOUND, torch_device):
        Nx = len(x) - 1
        Ny = len(y) - 1
        Nz = len(z) - 1
        dx = np.abs(np.diff(x)).reshape(1, 1, Nx)
        dy = np.abs(np.diff(y)).reshape(1, Ny, 1)
        dz = np.abs(np.diff(z)).reshape(Nz, 1, 1)

        self.SHP = (Nz, Ny, Nx)
        self.Nod = np.prod(self.SHP)
        self.active = (IBOUND > 0).reshape(self.Nod, )  # boolean vector denoting the active cells
        self.inact = (IBOUND == 0).reshape(self.Nod, )  # boolean vector denoting inacive cells
        self.fxhd = (IBOUND < 0).reshape(self.Nod, )  # boolean vector denoting fixed-head cells

        self.Rx_buffer = torch.as_tensor(0.5 * dx / (dy * dz), device=torch_device)
        self.Ry_buffer = torch.as_tensor(0.5 * dy / (dz * dx), device=torch_device)
        self.Rz_buffer = torch.as_tensor(0.5 * dz / (dx * dy), device=torch_device)

        NOD = np.arange(self.Nod).reshape(self.SHP)

        IW = NOD[:, :, :-1]  # west neighbor cell numbers
        IE = NOD[:, :, 1:]  # east neighbor cell numbers
        IS = NOD[:, :-1, :]  # south neighbor cell numbers
        IN = NOD[:, 1:, :]  # north neighbor cell numbers
        IT = NOD[:-1, :, :]  # top neighbor cell numbers
        IB = NOD[1:, :, :]  # bottom neighbor cell numbers

        R = lambda x: x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()

        idx_data = np.stack((np.concatenate((R(IE), R(IW), R(IN), R(IS), R(IB), R(IT))),
                             np.concatenate((R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)))))
        self.idx = torch.as_tensor(idx_data, device=torch_device)
        self.FQ = torch.as_tensor(FQ.reshape(self.Nod, 1), dtype=torch.float64, device=torch_device)
        self.IH = IH
        self.torch_device = torch_device

    def run(self, kd):
        Out = namedtuple('Out', ['Phi', 'Q', 'Qx', 'Qy', 'Qz'])

        Rx = self.Rx_buffer / kd
        Ry = self.Ry_buffer / kd
        Rz = self.Rz_buffer / kd

        # set flow resistance in inactive cells to infinite
        Rx = Rx.reshape(self.Nod)
        Rx[self.inact] = torch.as_tensor(np.Inf)
        Rx = Rx.reshape(self.SHP)
        Ry = Ry.reshape(self.Nod)
        Ry[self.inact] = torch.as_tensor(np.Inf)
        Ry = Ry.reshape(self.SHP)
        Rz = Rz.reshape(self.Nod)
        Rz[self.inact] = torch.as_tensor(np.Inf)
        Rz = Rz.reshape(self.SHP)

        # conductances between adjacent cells
        Cx = 1 / (Rx[:, :, :-1] + Rx[:, :, 1:])
        Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])
        Cz = 1 / (Rz[:-1, :, :] + Rz[1:, :, :])

        vals = -1 * torch.cat(
            (torch.ravel(Cx), torch.ravel(Cx), torch.ravel(Cy), torch.ravel(Cy), torch.ravel(Cz), torch.ravel(Cz))
        )
        A = torch.sparse_coo_tensor(self.idx, vals, (self.Nod, self.Nod))
        adiag = torch.sparse.sum(-1 * A, dim=1).to_dense()
        A = A + torch.diag(adiag).to_sparse()
        A = A.coalesce()

        IH_buffer1 = torch.as_tensor(self.IH.copy().reshape(self.Nod, 1)[self.fxhd], dtype=torch.float64, device=self.torch_device)
        RHS = self.FQ - torch.mm(A.to_dense()[:, self.fxhd], IH_buffer1)

        Out.Phi = torch.as_tensor(self.IH.copy().flatten(), dtype=torch.float64, device=self.torch_device).reshape(self.Nod, 1)
        #Out.Phi[self.active] = solve(torch.unsqueeze(A.to_dense()[self.active][:, self.active], dim=0).to_sparse(), torch.unsqueeze(RHS[self.active], dim=0))[0]

        Out.Phi[self.active] = torch.unsqueeze(tzj_solve.sparse_solve(A.to_dense()[self.active][:, self.active].to_sparse(), torch.squeeze(RHS[self.active])), 1)

        # net cell inflow
        Out.Q = torch.sparse.mm(A, Out.Phi.reshape(self.Nod, 1)).reshape(self.SHP)

        Out.Q[:, :, 1:-1] = 0.0

        # set inactive cells to NaN, We need to comment out, because the PHI is modified, it will cause errors in backpropagation
        #Out.Phi[self.inact] = torch.as_tensor(np.NaN)  # put NaN at inactive locations

        # reshape Phi to shape of grid
        Out.Phi = Out.Phi.reshape(self.SHP)

        # Flows across cell faces
        Out.Qx = -1 * torch.diff(Out.Phi, dim=2) * Cx
        Out.Qy = -1 * torch.diff(Out.Phi, dim=1) * Cy
        Out.Qz = -1 * torch.diff(Out.Phi, dim=0) * Cz




        return Out  # all outputs in a named tuple for easy access




if __name__ == "__main__":
    torch_device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     torch_device = torch.device('cuda')
    #     print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
    print('Running on ' + str(torch_device))

    # from paper
    input_file = "./paper_data/input_999.hdf5"

    f = h5py.File(input_file, "r")
    kd = np.array(f["kd"])
    kd = np.exp(kd)
    spd = np.array(f["welspd"])
    spd = {
        i: [
            tuple(
                [int(spd[i, 0]), int(spd[i, 1]), int(spd[i, 2]),
                 spd[i, 3],
                 int(spd[i, 4])]
            )
        ]
        for i in range(len(spd))
    }
    f.close()

    # specify a rectangular grid
    x = np.arange(0., 2500., 30.86)
    y = np.arange(0., 1250., 30.48)  # backward, i.e. first row grid line has highest y
    z = np.arange(0., 301., 50.)  # backward, i.e. from top to bottom

    SHP = (len(z) - 1, len(y) - 1, len(x) - 1)

    FQ = np.zeros(SHP)  # all flows zero. Note sz is the shape of the model grid

    IH = np.zeros(SHP, dtype=np.float64)
    h_grad = 0.012
    l_head, r_head = h_grad * 2500, 0.
    IH[:, :, 0] = l_head
    IH[:, :, -1] = r_head

    IBOUND = np.ones(SHP)
    IBOUND[:, :, 0] = -1
    IBOUND[:, :, -1] = -1

    M_FDM3 = ModflowFDM3(x, y, z, FQ, IH, IBOUND, torch_device)

    kd = torch.tensor(kd, dtype=torch.float64, requires_grad=True, device=torch_device)
    Out = M_FDM3.run(kd)

    # test the result using the Flopy result
    f = h5py.File("./paper_data/output_999.hdf5", "r")
    conc = np.array(f['concentration'])
    heads = np.array(f['head'])
    f.close()

    error_percentage = np.sum(Out.Phi.cpu().detach().numpy() - heads) / np.sum(Out.Phi.cpu().detach().numpy())

    print(error_percentage)

    # visualize the head and the flows
    # the center point coordinates
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])

    layer = 0  # contours for this layer
    nc = 50  # number of contours in total

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title("Contours (%d in total) of the head in layer %d with inactive section" % (nc, layer))
    plt.contour(xm, ym, Out.Phi[layer].cpu().detach().numpy(), nc)

    plt.show()












