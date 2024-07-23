# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 16:43
# @Author  : Zhenjie Tang
# @File    : MT3DMS_FDM3.py
# @Description: Computing contaminant concentration of steady-state flow using the FDM,
# like the MT3DMS (itype = -1, constant-concentration cell)

import numpy as np
import h5py
import Utils.DataVisualization as visul_tool
import torch
from GroundwaterModelPytorch import Modflow_FDM3_class
from torch_sparse_solve import solve
import sparse_linear_systems.sparse_solver as tzj_solve


class Mt3dmsFDM3(object):
    def __init__(self, x, y, z, t, ICBUND, IC, stress_period_step_size, transport_step_size,
         save_step_size, torch_device, al=0.01, trpt=0.1, trpv=0.01, dmcoef=1.e-9, prsity=0.25, epsilon=1.0):

        # Preparation before calculating coefficient matrix
        self.torch_device = torch_device
        self.al = al
        self.trpt = trpt
        self.trpv = trpv
        self.dmcoef = dmcoef
        self.prsity = prsity

        self.Nx = len(x) - 1
        self.Ny = len(y) - 1
        self.Nz = len(z) - 1
        self.dx = np.abs(np.diff(x)).reshape(1, 1, self.Nx)
        self.dy = np.abs(np.diff(y)).reshape(1, self.Ny, 1)
        self.dz = np.abs(np.diff(z)).reshape(self.Nz, 1, 1)
        self.SHP = (self.Nz, self.Ny, self.Nx)
        self.Nod = np.prod(self.SHP)

        NOD = np.arange(self.Nod).reshape(self.SHP)
        IE = NOD[:, :, 1:]  # east neighbor cell numbers
        IW = NOD[:, :, :-1]  # west neighbor cell numbers
        IN = NOD[:, 1:, :]  # north neighbor cell numbers
        IS = NOD[:, :-1, :]  # south neighbor cell numbers
        IB = NOD[1:, :, :]  # bottom neighbor cell numbers
        IT = NOD[:-1, :, :]  # top neighbor cell numbers
        R = lambda x: x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()

        self.alphaW = torch.as_tensor(self.dx[:, :, 1:] / (self.dx[:, :, :-1] + self.dx[:, :, 1:]), device=torch_device)
        self.alphaE = torch.as_tensor(self.dx[:, :, :-1] / (self.dx[:, :, :-1] + self.dx[:, :, 1:]), device=torch_device)
        self.alphaS = torch.as_tensor(self.dy[:, 1:, :] / (self.dy[:, :-1, :] + self.dy[:, 1:, :]), device=torch_device)
        self.alphaN = torch.as_tensor(self.dy[:, :-1, :] / (self.dy[:, :-1, :] + self.dy[:, 1:, :]), device=torch_device)
        self.alphaT = torch.as_tensor(self.dz[1:, :, :] / (self.dz[:-1, :, :] + self.dz[1:, :, :]), device=torch_device)
        self.alphaB = torch.as_tensor(self.dz[:-1, :, :] / (self.dz[:-1, :, :] + self.dz[1:, :, :]), device=torch_device)

        idx_data = np.stack((np.concatenate((R(IE), R(IW), R(IN), R(IS), R(IT), R(IB))),
                             np.concatenate((R(IW), R(IE), R(IS), R(IN), R(IB), R(IT)))
                             ))
        self.idx = torch.as_tensor(idx_data, device=torch_device)

        # generate a coefficient matrix of diffusion x, y, z direction
        self.xx_diff_buffer = torch.as_tensor(self.dy * self.dz / (0.5 * (self.dx[:, :, :-1] + self.dx[:, :, 1:])), device=torch_device)
        self.yy_diff_buffer = torch.as_tensor(self.dx * self.dz / (0.5 * (self.dy[:, :-1, :] + self.dy[:, 1:, :])), device=torch_device)
        self.zz_diff_buffer = torch.as_tensor(self.dx * self.dy / (0.5 * (self.dz[:-1, :, :] + self.dz[1:, :, :])), device=torch_device)

        # generate a coefficient matrix of diffusion xy
        self.C_xy_temp = torch.as_tensor(self.dy[:, 1:-1, :] * self.dz / (self.dy[:, 1:-1, :] + 0.5 * (self.dy[:, 2:, :] + self.dy[:, :-2, :])), device=torch_device)
        self.C_x_buffer1 = torch.as_tensor(self.dx[:, :, 2:] / (self.dx[:, :, 1:-1] + self.dx[:, :, 2:]), device=torch_device)
        self.C_x_buffer2 = torch.as_tensor(self.dx[:, :, 1:-1] / (self.dx[:, :, 1:-1] + self.dx[:, :, 2:]), device=torch_device)
        self.C_x_buffer3 = torch.as_tensor(self.dx[:, :, 2:] / (self.dx[:, :, 1:-1] + self.dx[:, :, 2:]), device=torch_device)
        self.C_x_buffer4 = torch.as_tensor(self.dx[:, :, 1:-1] / (self.dx[:, :, 1:-1] + self.dx[:, :, 2:]), device=torch_device)
        self.C_x_buffer5 = torch.as_tensor(self.dx[:, :, 1:-1] / (self.dx[:, :, 1:-1] + self.dx[:, :, :-2]), device=torch_device)
        self.C_x_buffer6 = torch.as_tensor(self.dx[:, :, :-2] / (self.dx[:, :, 1:-1] + self.dx[:, :, :-2]), device=torch_device)
        self.C_x_buffer7 = torch.as_tensor(self.dx[:, :, 1:-1] / (self.dx[:, :, 1:-1] + self.dx[:, :, :-2]), device=torch_device)
        self.C_x_buffer8 = torch.as_tensor(self.dx[:, :, :-2] / (self.dx[:, :, 1:-1] + self.dx[:, :, :-2]), device=torch_device)
        xy_start = NOD[:, 1:-1, 1:-1]
        xy_end1 = NOD[:, :-2, :-2]
        xy_end2 = NOD[:, :-2, 1:-1]
        xy_end3 = NOD[:, :-2, 2:]
        xy_end4 = NOD[:, 2:, :-2]
        xy_end5 = NOD[:, 2:, 1:-1]
        xy_end6 = NOD[:, 2:, 2:]

        idx_data_xy = np.stack(
            (np.concatenate((R(xy_start), R(xy_start), R(xy_start), R(xy_start), R(xy_start), R(xy_start))),
             np.concatenate((R(xy_end1), R(xy_end2), R(xy_end3), R(xy_end4), R(xy_end5), R(xy_end6)))
             ))
        self.idx_xy = torch.as_tensor(idx_data_xy, device=torch_device)

        # generate a coefficient matrix of diffusion xz
        self.C_xz_temp = torch.as_tensor(self.dy * self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + 0.5 * (self.dz[2:, :, :] + self.dz[:-2, :, :])), device=torch_device)

        xz_start = NOD[1:-1, :, 1:-1]
        xz_end1 = NOD[:-2, :, :-2]
        xz_end2 = NOD[:-2, :, 1:-1]
        xz_end3 = NOD[:-2, :, 2:]
        xz_end4 = NOD[2:, :, :-2]
        xz_end5 = NOD[2:, :, 1:-1]
        xz_end6 = NOD[2:, :, 2:]

        idx_data_xz = np.stack(
            (np.concatenate((R(xz_start), R(xz_start), R(xz_start), R(xz_start), R(xz_start), R(xz_start))),
             np.concatenate((R(xz_end1), R(xz_end2), R(xz_end3), R(xz_end4), R(xz_end5), R(xz_end6)))
             ))
        self.idx_xz = torch.as_tensor(idx_data_xz, device=torch_device)

        # generate a coefficient matrix of diffusion yx
        self.C_yx_temp = torch.as_tensor(self.dx[:, :, 1:-1] * self.dz / (self.dx[:, :, 1:-1] + 0.5 * (self.dx[:, :, 2:] + self.dx[:, :, :-2])), device=torch_device)
        self.C_y_buffer1 = torch.as_tensor(self.dy[:, 2:, :] / (self.dy[:, 1:-1, :] + self.dy[:, 2:, :]), device=torch_device)
        self.C_y_buffer2 = torch.as_tensor(self.dy[:, 1:-1, :] / (self.dy[:, 1:-1, :] + self.dy[:, 2:, :]), device=torch_device)
        self.C_y_buffer3 = torch.as_tensor(self.dy[:, 2:, :] / (self.dy[:, 1:-1, :] + self.dy[:, 2:, :]), device=torch_device)
        self.C_y_buffer4 = torch.as_tensor(self.dy[:, 1:-1, :] / (self.dy[:, 1:-1, :] + self.dy[:, 2:, :]), device=torch_device)
        self.C_y_buffer5 = torch.as_tensor(self.dy[:, 1:-1, :] / (self.dy[:, 1:-1, :] + self.dy[:, :-2, :]), device=torch_device)
        self.C_y_buffer6 = torch.as_tensor(self.dy[:, :-2, :] / (self.dy[:, 1:-1, :] + self.dy[:, :-2, :]), device=torch_device)
        self.C_y_buffer7 = torch.as_tensor(self.dy[:, 1:-1, :] / (self.dy[:, 1:-1, :] + self.dy[:, :-2, :]), device=torch_device)
        self.C_y_buffer8 = torch.as_tensor(self.dy[:, :-2, :] / (self.dy[:, 1:-1, :] + self.dy[:, :-2, :]), device=torch_device)

        yx_start = NOD[:, 1:-1, 1:-1]
        yx_end1 = NOD[:, :-2, :-2]
        yx_end2 = NOD[:, 1:-1, :-2]
        yx_end3 = NOD[:, 2:, :-2]
        yx_end4 = NOD[:, :-2, 2:]
        yx_end5 = NOD[:, 1:-1, 2:]
        yx_end6 = NOD[:, 2:, 2:]

        idx_data_yx = np.stack(
            (np.concatenate((R(yx_start), R(yx_start), R(yx_start), R(yx_start), R(yx_start), R(yx_start))),
             np.concatenate((R(yx_end1), R(yx_end2), R(yx_end3), R(yx_end4), R(yx_end5), R(yx_end6)))
             ))
        self.idx_yx = torch.as_tensor(idx_data_yx, device=torch_device)

        # generate a coefficient matrix of diffusion yz
        self.C_yz_temp = torch.as_tensor(self.dx * self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + 0.5 * (self.dz[2:, :, :] + self.dz[:-2, :, :])), device=torch_device)

        yz_start = NOD[1:-1, 1:-1, :]
        yz_end1 = NOD[:-2, :-2, :]
        yz_end2 = NOD[:-2, 1:-1, :]
        yz_end3 = NOD[:-2, 2:, :]
        yz_end4 = NOD[2:, :-2, :]
        yz_end5 = NOD[2:, 1:-1, :]
        yz_end6 = NOD[2:, 2:, :]

        idx_data_yz = np.stack(
            (np.concatenate((R(yz_start), R(yz_start), R(yz_start), R(yz_start), R(yz_start), R(yz_start))),
             np.concatenate((R(yz_end1), R(yz_end2), R(yz_end3), R(yz_end4), R(yz_end5), R(yz_end6)))
             ))
        self.idx_yz = torch.as_tensor(idx_data_yz, device=torch_device)

        # generate a coefficient matrix of diffusion zx
        self.C_zx_temp = torch.as_tensor(self.dx[:, :, 1:-1] * self.dy / (self.dx[:, :, 1:-1] + 0.5 * (self.dx[:, :, 2:] + self.dx[:, :, :-2])), device=torch_device)
        self.C_z_buffer1 = torch.as_tensor(self.dz[2:, :, :] / (self.dz[1:-1, :, :] + self.dz[2:, :, :]), device=torch_device)
        self.C_z_buffer2 = torch.as_tensor(self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + self.dz[2:, :, :]), device=torch_device)
        self.C_z_buffer3 = torch.as_tensor(self.dz[2:, :, :] / (self.dz[1:-1, :, :] + self.dz[2:, :, :]), device=torch_device)
        self.C_z_buffer4 = torch.as_tensor(self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + self.dz[2:, :, :]), device=torch_device)
        self.C_z_buffer5 = torch.as_tensor(self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + self.dz[:-2, :, :]), device=torch_device)
        self.C_z_buffer6 = torch.as_tensor(self.dz[:-2, :, :] / (self.dz[1:-1, :, :] + self.dz[:-2, :, :]), device=torch_device)
        self.C_z_buffer7 = torch.as_tensor(self.dz[1:-1, :, :] / (self.dz[1:-1, :, :] + self.dz[:-2, :, :]), device=torch_device)
        self.C_z_buffer8 = torch.as_tensor(self.dz[:-2, :, :] / (self.dz[1:-1, :, :] + self.dz[:-2, :, :]), device=torch_device)

        zx_start = NOD[1:-1, :, 1:-1]
        zx_end1 = NOD[:-2, :, :-2]
        zx_end2 = NOD[1:-1, :, :-2]
        zx_end3 = NOD[2:, :, :-2]
        zx_end4 = NOD[:-2, :, 2:]
        zx_end5 = NOD[1:-1, :, 2:]
        zx_end6 = NOD[2:, :, 2:]

        idx_data_zx = np.stack(
            (np.concatenate((R(zx_start), R(zx_start), R(zx_start), R(zx_start), R(zx_start), R(zx_start))),
             np.concatenate((R(zx_end1), R(zx_end2), R(zx_end3), R(zx_end4), R(zx_end5), R(zx_end6)))
             ))
        self.idx_zx = torch.as_tensor(idx_data_zx, device=torch_device)

        # generate a coefficient matrix of diffusion zy
        self.C_zy_temp = torch.as_tensor(self.dx * self.dy[:, 1:-1, ] / (self.dy[:, 1:-1, :] + 0.5 * (self.dy[:, 2:, :] + self.dy[:, :-2, :])), device=torch_device)

        zy_start = NOD[1:-1, 1:-1, :]
        zy_end1 = NOD[:-2, :-2, :]
        zy_end2 = NOD[1:-1, :-2, :]
        zy_end3 = NOD[2:, :-2, :]
        zy_end4 = NOD[:-2, 2:, :]
        zy_end5 = NOD[1:-1, 2:, :]
        zy_end6 = NOD[2:, 2:, :]

        idx_data_zy = np.stack(
            (np.concatenate((R(zy_start), R(zy_start), R(zy_start), R(zy_start), R(zy_start), R(zy_start))),
             np.concatenate((R(zy_end1), R(zy_end2), R(zy_end3), R(zy_end4), R(zy_end5), R(zy_end6)))
             ))
        self.idx_zy = torch.as_tensor(idx_data_zy, device=torch_device)

        # run
        self.t = t
        self.transport_step_size = transport_step_size
        self.save_step_size = save_step_size
        self.ICBUND = ICBUND
        self.IC = IC
        self.stress_period_step_size = stress_period_step_size
        self.epsilon = epsilon

        rhob = 1.587  # bulk density of porous media g/m^3
        kd = 0.1  # the distribution coefficient m^3/g

        R = 1 + rhob / prsity * kd
        V = self.dx * self.dy * self.dz

        self.adiag_aux = torch.as_tensor((-1) * ((R * prsity * V) / (epsilon * transport_step_size)).ravel(), device=torch_device)

    def getCmatrix_Convection(self, Out):
        """ Preparation """
        Qx, Qy, Qz = Out.Qx, Out.Qy, Out.Qz

        """ Calculation coefficient """
        Cx_W = self.alphaW * Qx
        Cx_E = self.alphaE * (-1 * Qx)
        Cy_S = self.alphaS * Qy
        Cy_N = self.alphaN * (-1 * Qy)
        Cz_T = self.alphaT * Qz
        Cz_B = self.alphaB * (-1 * Qz)

        vals = torch.cat(
            (torch.ravel(Cx_W), torch.ravel(Cx_E), torch.ravel(Cy_S),
             torch.ravel(Cy_N), torch.ravel(Cz_B), torch.ravel(Cz_T))
        )

        A_conv = torch.sparse_coo_tensor(self.idx, vals, (self.Nod, self.Nod))
        A_conv = A_conv.coalesce()

        Cx_W_end = (1 - self.alphaW) * Qx
        Cx_E_end = (1 - self.alphaE) * (-1 * Qx)

        Cy_S_end = (1 - self.alphaS) * Qy
        Cy_N_end = (1 - self.alphaN) * (-1 * Qy)

        Cz_T_end = (1 - self.alphaT) * Qz
        Cz_B_end = (1 - self.alphaB) * (-1 * Qz)

        vals_end = torch.cat(
            (torch.ravel(Cx_W_end), torch.ravel(Cx_E_end), torch.ravel(Cy_S_end),
             torch.ravel(Cy_N_end), torch.ravel(Cz_B_end), torch.ravel(Cz_T_end))
        )

        A_conv_end = torch.sparse_coo_tensor(self.idx, vals_end, (self.Nod, self.Nod))
        A_conv_end = A_conv_end.coalesce()

        A_conv_adiag1 = torch.sparse.sum(A_conv_end, dim=1).to_dense()
        A_conv_adiag2 = Out.Q.reshape(self.Nod, )
        A_conv_adiag2[(Out.Q > 0).reshape(self.Nod, )] = 0
        A_conv_adiag = A_conv_adiag1 + A_conv_adiag2

        A_conv = A_conv + torch.diag(A_conv_adiag).to_sparse()
        A_conv = A_conv.coalesce()

        return A_conv

    def getDiffusionCoefficient(self, Out):
        Qx, Qy, Qz = Out.Qx, Out.Qy, Out.Qz
        # compute the velocity components for the central coordinates
        Vx = torch.cat((Qx[:, :, 0].reshape((self.Nz, self.Ny, 1)) * 0.5,
                                0.5 * (Qx[:, :, :-1].reshape((self.Nz, self.Ny, self.Nx - 2)) +
                                Qx[:, :, 1:].reshape((self.Nz, self.Ny, self.Nx - 2))),
                                Qx[:, :, -1].reshape((self.Nz, self.Ny, 1)) * 0.5), dim=2).reshape((self.Nz, self.Ny, self.Nx))
        Vx = Vx / torch.as_tensor((self.dy * self.dz) * self.prsity, device=self.torch_device)

        Vy = torch.cat((Qy[:, 0, :].reshape((self.Nz, 1, self.Nx)) * 0.5,
                        0.5 * (Qy[:, :-1, :].reshape((self.Nz, self.Ny - 2, self.Nx)) +
                               Qy[:, 1:, :].reshape((self.Nz, self.Ny - 2, self.Nx))),
                        Qy[:, -1, :].reshape((self.Nz, 1, self.Nx)) * 0.5), dim=1).reshape((self.Nz, self.Ny, self.Nx))
        Vy = Vy / torch.as_tensor((self.dx * self.dz) * self.prsity, device=self.torch_device)

        Vz = torch.cat((Qz[0, :, :].reshape((1, self.Ny, self.Nx)) * 0.5,
                        0.5 * (Qz[:-1, :, :].reshape((self.Nz - 2, self.Ny, self.Nx)) +
                               Qz[1:, :, :].reshape((self.Nz - 2, self.Ny, self.Nx))),
                        Qz[-1, :, :].reshape((1, self.Ny, self.Nx)) * 0.5), dim=0).reshape((self.Nz, self.Ny, self.Nx))
        Vz = Vz / torch.as_tensor((self.dx * self.dy) * self.prsity, device=self.torch_device)

        # generate dispersion coefficient matrix (L2/T), like the MT3DMS
        V = torch.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
        D_Vx = (Vx * Vx) / V
        D_Vy = (Vy * Vy) / V
        D_Vz = (Vz * Vz) / V

        ##### 如果考虑inactive cell 只需要对下面D开头的进行修改就好，因为这些是通量。
        D_xx = (self.al * D_Vx + self.al * self.trpt * D_Vy + self.al * self.trpv * D_Vz + self.dmcoef) * self.prsity
        D_yy = (self.al * D_Vy + self.al * self.trpt * D_Vx + self.al * self.trpv * D_Vz + self.dmcoef) * self.prsity
        D_zz = (self.al * D_Vz + self.al * self.trpv * D_Vx + self.al * self.trpv * D_Vy + self.dmcoef) * self.prsity
        D_xy = D_yx = (self.al - self.al * self.trpt) * Vx * Vy / V * self.prsity
        D_xz = D_zx = (self.al - self.al * self.trpv) * Vx * Vz / V * self.prsity
        D_yz = D_zy = (self.al - self.al * self.trpv) * Vy * Vz / V * self.prsity

        return D_xx, D_yy, D_zz, D_xy, D_xz, D_yx, D_yz, D_zx, D_zy

    def getCmatrix_diffusion(self, D_xx, D_yy, D_zz):
        ##########  generate a coefficient matrix of diffusion x, y, z direction for the mass conservation equation #####
        # compute dispersion coefficient value at a cell interface
        D_xx_interface = D_xx[:, :, :-1] * self.alphaW + D_xx[:, :, 1:] * self.alphaE

        D_yy_interface = D_yy[:, :-1, :] * self.alphaS + D_yy[:, 1:, :] * self.alphaN

        D_zz_interface = D_zz[:-1, :, :] * self.alphaT + D_zz[1:, :, :] * self.alphaB

        C_xx_diff = D_xx_interface * self.xx_diff_buffer
        C_yy_diff = D_yy_interface * self.yy_diff_buffer
        C_zz_diff = D_zz_interface * self.zz_diff_buffer

        vals_diff = torch.cat(
            (torch.ravel(C_xx_diff), torch.ravel(C_xx_diff), torch.ravel(C_yy_diff),
             torch.ravel(C_yy_diff), torch.ravel(C_zz_diff), torch.ravel(C_zz_diff))
        )

        A_diff = torch.sparse_coo_tensor(self.idx, vals_diff, (self.Nod, self.Nod))

        A_diff_Adiag = torch.sparse.sum(-1 * A_diff, dim=1).to_dense()
        A_diff = A_diff + torch.diag(A_diff_Adiag).to_sparse()
        A_diff = A_diff.coalesce()

        return A_diff

    def getCmatrix_diffusion_xy(self, D_xy):
        D_xy_interface = D_xy[:, :, :-1] * self.alphaW + D_xy[:, :, 1:] * self.alphaE

        C_xy_right_top_left = D_xy_interface[:, 1:-1, 1:] * self.C_xy_temp * self.C_x_buffer1
        C_xy_right_top_right = D_xy_interface[:, 1:-1, 1:] * self.C_xy_temp * self.C_x_buffer2
        C_xy_right_down_left = D_xy_interface[:, 1:-1, 1:] * (-1) * self.C_xy_temp * self.C_x_buffer3
        C_xy_right_down_right = D_xy_interface[:, 1:-1, 1:] * self.C_xy_temp * (-1) * self.C_x_buffer4

        C_xy_left_top_left = (-1) * D_xy_interface[:, 1:-1, :-1] * self.C_xy_temp * self.C_x_buffer5
        C_xy_left_top_right = (-1) * D_xy_interface[:, 1:-1, :-1] * self.C_xy_temp * self.C_x_buffer6
        C_xy_left_down_left = (-1) * D_xy_interface[:, 1:-1, :-1] * self.C_xy_temp * (-1) * self.C_x_buffer7
        C_xy_left_down_right = (-1) * D_xy_interface[:, 1:-1, :-1] * self.C_xy_temp * (-1) * self.C_x_buffer8



        vals_diff_xy = torch.cat(
            (torch.ravel(C_xy_left_top_left), torch.ravel(C_xy_right_top_left + C_xy_left_top_right),
             torch.ravel(C_xy_right_top_right),
             torch.ravel(C_xy_left_down_left), torch.ravel(C_xy_right_down_left + C_xy_left_down_right),
             torch.ravel(C_xy_right_down_right))
        )
        A_diff_xy = torch.sparse_coo_tensor(self.idx_xy, vals_diff_xy, (self.Nod, self.Nod))

        return A_diff_xy

    def getCmatrix_diffusion_xz(self, D_xz):
        D_xz_interface = D_xz[:, :, :-1] * self.alphaW + D_xz[:, :, 1:] * self.alphaE

        C_xz_right_top_left = D_xz_interface[1:-1, :, 1:] * self.C_xz_temp * self.C_x_buffer1
        C_xz_right_top_right = D_xz_interface[1:-1, :, 1:] * self.C_xz_temp * self.C_x_buffer2
        C_xz_right_down_left = D_xz_interface[1:-1, :, 1:] * (-1) * self.C_xz_temp * self.C_x_buffer3
        C_xz_right_down_right = D_xz_interface[1:-1, :, 1:] * self.C_xz_temp * (-1) * self.C_x_buffer4

        C_xz_left_top_left = (-1) * D_xz_interface[1:-1, :, :-1] * self.C_xz_temp * self.C_x_buffer5
        C_xz_left_top_right = (-1) * D_xz_interface[1:-1, :, :-1] * self.C_xz_temp * self.C_x_buffer6
        C_xz_left_down_left = (-1) * D_xz_interface[1:-1, :, :-1] * self.C_xz_temp * (-1) * self.C_x_buffer7
        C_xz_left_down_right = (-1) * D_xz_interface[1:-1, :, :-1] * self.C_xz_temp * (-1) * self.C_x_buffer8

        vals_diff_xz = torch.cat(
            (torch.ravel(C_xz_left_top_left), torch.ravel(C_xz_right_top_left + C_xz_left_top_right),
             torch.ravel(C_xz_right_top_right),
             torch.ravel(C_xz_left_down_left), torch.ravel(C_xz_right_down_left + C_xz_left_down_right),
             torch.ravel(C_xz_right_down_right))
        )

        A_diff_xz = torch.sparse_coo_tensor(self.idx_xz, vals_diff_xz, (self.Nod, self.Nod))

        return A_diff_xz

    def getCmatrix_diffusion_yx(self, D_yx):
        D_yx_interface = D_yx[:, :-1, :] * self.alphaS + D_yx[:, 1:, :] * self.alphaN


        C_yx_right_top_left = D_yx_interface[:, 1:, 1:-1] * self.C_yx_temp * self.C_y_buffer1
        C_yx_right_top_right = D_yx_interface[:, 1:, 1:-1] * self.C_yx_temp * self.C_y_buffer2
        C_yx_right_down_left = D_yx_interface[:, 1:, 1:-1] * (-1) * self.C_yx_temp * self.C_y_buffer3
        C_yx_right_down_right = D_yx_interface[:, 1:, 1:-1] * self.C_yx_temp * (-1) * self.C_y_buffer4

        C_yx_left_top_left = (-1) * D_yx_interface[:, :-1, 1:-1] * self.C_yx_temp * self.C_y_buffer5
        C_yx_left_top_right = (-1) * D_yx_interface[:, :-1, 1:-1] * self.C_yx_temp * self.C_y_buffer6
        C_yx_left_down_left = (-1) * D_yx_interface[:, :-1, 1:-1] * self.C_yx_temp * (-1) * self.C_y_buffer7
        C_yx_left_down_right = (-1) * D_yx_interface[:, :-1, 1:-1] * self.C_yx_temp * (-1) * self.C_y_buffer8


        vals_diff_yx = torch.cat(
            (torch.ravel(C_yx_left_top_left), torch.ravel(C_yx_right_top_left + C_yx_left_top_right),
             torch.ravel(C_yx_right_top_right),
             torch.ravel(C_yx_left_down_left), torch.ravel(C_yx_right_down_left + C_yx_left_down_right),
             torch.ravel(C_yx_right_down_right))
        )

        A_diff_yx = torch.sparse_coo_tensor(self.idx_yx, vals_diff_yx, (self.Nod, self.Nod))

        return A_diff_yx

    def getCmatrix_diffusion_yz(self, D_yz):
        D_yz_interface = D_yz[:, :-1, :] * self.alphaS + D_yz[:, 1:, :] * self.alphaN


        C_yz_right_top_left = D_yz_interface[1:-1, 1:, :] * self.C_yz_temp * self.C_y_buffer1
        C_yz_right_top_right = D_yz_interface[1:-1, 1:, :] * self.C_yz_temp * self.C_y_buffer2
        C_yz_right_down_left = D_yz_interface[1:-1, 1:, :] * (-1) * self.C_yz_temp * self.C_y_buffer3
        C_yz_right_down_right = D_yz_interface[1:-1, 1:, :] * self.C_yz_temp * (-1) * self.C_y_buffer4

        C_yz_left_top_left = (-1) * D_yz_interface[1:-1, :-1, :] * self.C_yz_temp * self.C_y_buffer5
        C_yz_left_top_right = (-1) * D_yz_interface[1:-1, :-1, :] * self.C_yz_temp * self.C_y_buffer6
        C_yz_left_down_left = (-1) * D_yz_interface[1:-1, :-1, :] * self.C_yz_temp * (-1) * self.C_y_buffer7
        C_yz_left_down_right = (-1) * D_yz_interface[1:-1, :-1, :] * self.C_yz_temp * (-1) * self.C_y_buffer8


        vals_diff_yz = torch.cat(
            (torch.ravel(C_yz_left_top_left), torch.ravel(C_yz_right_top_left + C_yz_left_top_right),
             torch.ravel(C_yz_right_top_right),
             torch.ravel(C_yz_left_down_left), torch.ravel(C_yz_right_down_left + C_yz_left_down_right),
             torch.ravel(C_yz_right_down_right))
        )
        A_diff_yz = torch.sparse_coo_tensor(self.idx_yz, vals_diff_yz, (self.Nod, self.Nod))

        return A_diff_yz

    def getCmatrix_diffusion_zx(self, D_zx):
        D_zx_interface = D_zx[:-1, :, :] * self.alphaT + D_zx[1:, :, :] * self.alphaB


        C_zx_right_top_left = D_zx_interface[1:, :, 1:-1] * self.C_zx_temp * self.C_z_buffer1
        C_zx_right_top_right = D_zx_interface[1:, :, 1:-1] * self.C_zx_temp * self.C_z_buffer2
        C_zx_right_down_left = D_zx_interface[1:, :, 1:-1] * (-1) * self.C_zx_temp * self.C_z_buffer3
        C_zx_right_down_right = D_zx_interface[1:, :, 1:-1] * self.C_zx_temp * (-1) * self.C_z_buffer4

        C_zx_left_top_left = (-1) * D_zx_interface[:-1, :, 1:-1] * self.C_zx_temp * self.C_z_buffer5
        C_zx_left_top_right = (-1) * D_zx_interface[:-1, :, 1:-1] * self.C_zx_temp * self.C_z_buffer6
        C_zx_left_down_left = (-1) * D_zx_interface[:-1, :, 1:-1] * self.C_zx_temp * (-1) * self.C_z_buffer7
        C_zx_left_down_right = (-1) * D_zx_interface[:-1, :, 1:-1] * self.C_zx_temp * (-1) * self.C_z_buffer8


        vals_diff_zx = torch.cat(
            (torch.ravel(C_zx_left_top_left), torch.ravel(C_zx_right_top_left + C_zx_left_top_right),
             torch.ravel(C_zx_right_top_right),
             torch.ravel(C_zx_left_down_left), torch.ravel(C_zx_right_down_left + C_zx_left_down_right),
             torch.ravel(C_zx_right_down_right))
        )
        A_diff_zx = torch.sparse_coo_tensor(self.idx_zx, vals_diff_zx, (self.Nod, self.Nod))

        return A_diff_zx

    def getCmatrix_diffusion_zy(self, D_zy):
        D_zy_interface = D_zy[:-1, :, :] * self.alphaT + D_zy[1:, :, :] * self.alphaB


        C_zy_right_top_left = D_zy_interface[1:, 1:-1, :] * self.C_zy_temp * self.C_z_buffer1
        C_zy_right_top_right = D_zy_interface[1:, 1:-1, :] * self.C_zy_temp * self.C_z_buffer2
        C_zy_right_down_left = D_zy_interface[1:, 1:-1, :] * (-1) * self.C_zy_temp * self.C_z_buffer3
        C_zy_right_down_right = D_zy_interface[1:, 1:-1, :] * self.C_zy_temp * (-1) * self.C_z_buffer4

        C_zy_left_top_left = (-1) * D_zy_interface[:-1, 1:-1, :] * self.C_zy_temp * self.C_z_buffer5
        C_zy_left_top_right = (-1) * D_zy_interface[:-1, 1:-1, :] * self.C_zy_temp * self.C_z_buffer6
        C_zy_left_down_left = (-1) * D_zy_interface[:-1, 1:-1, :] * self.C_zy_temp * (-1) * self.C_z_buffer7
        C_zy_left_down_right = (-1) * D_zy_interface[:-1, 1:-1, :] * self.C_zy_temp * (-1) * self.C_z_buffer8


        vals_diff_zy = torch.cat(
            (torch.ravel(C_zy_left_top_left), torch.ravel(C_zy_right_top_left + C_zy_left_top_right),
             torch.ravel(C_zy_right_top_right),
             torch.ravel(C_zy_left_down_left), torch.ravel(C_zy_right_down_left + C_zy_left_down_right),
             torch.ravel(C_zy_right_down_right))
        )
        A_diff_zy = torch.sparse_coo_tensor(self.idx_zy, vals_diff_zy, (self.Nod, self.Nod))

        return A_diff_zy

    def run(self, Out, stress_period_data):
        A_conv = self.getCmatrix_Convection(Out)
        D_xx, D_yy, D_zz, D_xy, D_xz, D_yx, D_yz, D_zx, D_zy = self.getDiffusionCoefficient(Out)

        A_diff = self.getCmatrix_diffusion(D_xx, D_yy, D_zz)
        A_diff_xy = self.getCmatrix_diffusion_xy(D_xy)
        A_diff_xz = self.getCmatrix_diffusion_xz(D_xz)
        A_diff_yx = self.getCmatrix_diffusion_yx(D_yx)
        A_diff_yz = self.getCmatrix_diffusion_yz(D_yz)
        A_diff_zx = self.getCmatrix_diffusion_zx(D_zx)
        A_diff_zy = self.getCmatrix_diffusion_zy(D_zy)

        A_total = A_conv + A_diff + A_diff_xy + A_diff_xz + A_diff_yx + A_diff_yz + A_diff_zx + A_diff_zy

        """
            The chemical reactions considered represent sorption of the dissolved contaminant onto the solid surface of the
            porous media. We assume the system to be in local chemical equilibrium, that is, sorption to be much faster than 
            advection and dispersion.  The linear sorption isotherm is used 
            """


        A_total = A_total + torch.diag(self.adiag_aux).to_sparse()
        A_total = A_total.coalesce()

        out_buffer = torch.zeros((self.t // self.transport_step_size + 1, self.Nod), device=self.torch_device, dtype=torch.float64)
        out_buffer[0, :] = torch.as_tensor(self.IC.ravel(), device=self.torch_device)

        """ Iteration process"""
        for index in range(self.t // self.transport_step_size):

            ICBUND_update = self.ICBUND.copy()
            IC_update = torch.as_tensor(self.IC.copy(), device=self.torch_device)

            stress_period_index_max = len(stress_period_data) - 1
            stress_period_index = (index * self.transport_step_size) // self.stress_period_step_size

            if stress_period_index > stress_period_index_max:
                fxct = (ICBUND_update < 0).reshape(self.Nod, )
                fixed_concentration = IC_update.reshape(self.Nod, )
            else:
                z_index, y_index, x_index, value, _ = stress_period_data[stress_period_index][0]
                ICBUND_update[z_index, y_index, x_index] = -1
                IC_update[z_index, y_index, x_index] = value
                fxct = (ICBUND_update < 0).reshape(self.Nod, )
                fixed_concentration = IC_update.reshape(self.Nod, )

            FQ = (self.adiag_aux * out_buffer[index, :]).reshape(self.Nod, 1)
            RHS = FQ - torch.mm(A_total.to_dense()[:, fxct],
                                fixed_concentration.reshape(self.Nod, 1)[fxct])  # Right-hand side vector
            active = (ICBUND_update > 0).reshape(self.Nod, )
            #mean_concentration = solve(torch.unsqueeze(A_total.to_dense()[active][:, active], dim=0).to_sparse(), torch.unsqueeze(RHS[active], dim=0))[0]
            mean_concentration = torch.unsqueeze(tzj_solve.sparse_solve(A_total.to_dense()[active][:, active].to_sparse(), torch.squeeze(RHS[active])),1)

            out_buffer[index + 1, :][active] = out_buffer[index, :][active] + (
                        torch.squeeze(mean_concentration) - out_buffer[index, :][active]) / self.epsilon
            out_buffer[index + 1, :][fxct] = fixed_concentration[fxct]
            print(index)
        out = out_buffer[::(self.save_step_size // self.transport_step_size), :][1:, :]
        out = out.reshape((self.t // self.save_step_size,) + self.SHP)

        return out

    def run_parallel(self, Out, stress_period_data, n_positions):
        A_conv = self.getCmatrix_Convection(Out)
        D_xx, D_yy, D_zz, D_xy, D_xz, D_yx, D_yz, D_zx, D_zy = self.getDiffusionCoefficient(Out)

        A_diff = self.getCmatrix_diffusion(D_xx, D_yy, D_zz)
        A_diff_xy = self.getCmatrix_diffusion_xy(D_xy)
        A_diff_xz = self.getCmatrix_diffusion_xz(D_xz)
        A_diff_yx = self.getCmatrix_diffusion_yx(D_yx)
        A_diff_yz = self.getCmatrix_diffusion_yz(D_yz)
        A_diff_zx = self.getCmatrix_diffusion_zx(D_zx)
        A_diff_zy = self.getCmatrix_diffusion_zy(D_zy)

        A_total = A_conv + A_diff + A_diff_xy + A_diff_xz + A_diff_yx + A_diff_yz + A_diff_zx + A_diff_zy

        """
            The chemical reactions considered represent sorption of the dissolved contaminant onto the solid surface of the
            porous media. We assume the system to be in local chemical equilibrium, that is, sorption to be much faster than 
            advection and dispersion.  The linear sorption isotherm is used 
            """


        A_total = A_total + torch.diag(self.adiag_aux).to_sparse()
        A_total = A_total.coalesce()

        out_buffer = torch.zeros((self.t // self.transport_step_size + 1, n_positions, self.Nod), device=self.torch_device, dtype=torch.float64)
        out_buffer[0, :, :] = torch.as_tensor(self.IC.ravel(), device=self.torch_device).repeat(n_positions, 1)

        """ Iteration process"""
        for index in range(self.t // self.transport_step_size):
            stress_period_index_max = len(stress_period_data) - 1
            stress_period_index = (index * self.transport_step_size) // self.stress_period_step_size

            if stress_period_index > stress_period_index_max:
                ICBUND_update = self.ICBUND.copy()
                IC_update = self.IC.copy()
                fxct = (ICBUND_update < 0).reshape(self.Nod, )
                fixed_concentration = torch.as_tensor(IC_update.reshape(self.Nod, ), device=self.torch_device)

                active = (ICBUND_update > 0).reshape(self.Nod, )
                RHS_ALL = []
                for n_index in range(n_positions):
                    FQ = (self.adiag_aux * out_buffer[index, n_index, :]).reshape(self.Nod, 1)
                    RHS = FQ - torch.mm(A_total.to_dense()[:, fxct],
                                        fixed_concentration.reshape(self.Nod, 1)[fxct])  # Right-hand side vector
                    RHS = RHS[active]
                    RHS_ALL.append(RHS)
                RHS_ALL = torch.cat(RHS_ALL, 1)

                mean_concentration = tzj_solve.sparse_solve(A_total.to_dense()[active][:, active].to_sparse(), RHS_ALL).t()

                out_buffer[index + 1, :, :][:, active] = (out_buffer[index, :, :][:, active] + (mean_concentration - out_buffer[index, :, :][:, active]) / self.epsilon)
                out_buffer[index + 1, :, :][:, fxct] = fixed_concentration[fxct].repeat(n_positions, 1)
            else:
                fxct_ALL = []
                active_All = []
                fixed_concentration_ALL = []
                mean_concentration_tzj_all = []
                for n_index in range(n_positions):
                    ICBUND_update = self.ICBUND.copy()
                    IC_update = torch.as_tensor(self.IC.copy(), device=self.torch_device)

                    z_index, y_index, x_index, value, _ = stress_period_data[stress_period_index][n_index]
                    ICBUND_update[z_index, y_index, x_index] = -1
                    # ！！！！！！！！pay attention the value = [-1, 1], therefore this function is used for training.！！！
                    IC_update[z_index, y_index, x_index] = torch.abs(value) * 900 + 100
                    fxct = (ICBUND_update < 0).reshape(self.Nod, )
                    fixed_concentration = IC_update.reshape(self.Nod, )

                    FQ = (self.adiag_aux * out_buffer[index, n_index, :]).reshape(self.Nod, 1)
                    RHS = FQ - torch.mm(A_total.to_dense()[:, fxct],
                                        fixed_concentration.reshape(self.Nod, 1)[fxct])  # Right-hand side vector
                    active = (ICBUND_update > 0).reshape(self.Nod, )

                    fxct_ALL.append(fxct)
                    active_All.append(active)
                    fixed_concentration_ALL.append(fixed_concentration[fxct])
                    mean_concentration_tzj_all.append(tzj_solve.sparse_solve(A_total.to_dense()[active][:, active].to_sparse(), RHS[active]))

                mean_concentration_new = torch.stack(mean_concentration_tzj_all, dim=0)

                for n_index in range(n_positions):
                    out_buffer[index + 1, n_index, :][active_All[n_index]] = out_buffer[index, n_index, :][active_All[n_index]] + (
                            torch.squeeze(mean_concentration_new[n_index]) - out_buffer[index, n_index, :][active_All[n_index]]) / self.epsilon
                    out_buffer[index + 1, n_index, :][fxct_ALL[n_index]] = fixed_concentration_ALL[n_index]
            print(index)
        out = out_buffer[::(self.save_step_size // self.transport_step_size), :, :][1:, :, :]
        out = out.reshape((self.t // self.save_step_size, n_positions) + self.SHP)

        return out


if __name__ == "__main__":
    torch_device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     torch_device = torch.device('cuda')
    #     print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
    print('Running on ' + str(torch_device))

    # from paper
    input_file = "./paper_data/input_3.hdf5"

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
    x = np.linspace(0, 2500, 82)
    y = np.linspace(0, 1250, 42)
    z = np.linspace(0, 300, 7)

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

    # run the Steady state 3D Finite Difference Model
    M_FDM3 = Modflow_FDM3_class.ModflowFDM3(x, y, z, FQ, IH, IBOUND, torch_device)

    kd = torch.tensor(kd, dtype=torch.float64, requires_grad=True, device=torch_device)
    Out = M_FDM3.run(kd)

    IC = np.zeros(SHP)
    ICBUND = np.ones(SHP)
    total_days = 40 * 365  # days
    transport_step_size = 1 * 365  # days
    save_step_size = 4 * 365
    stress_period_step_size = 4 * 365
    prsity = 0.3
    al = 35.  # meter
    trpt = 0.3
    trpv = 0.3
    del spd[5]

    for key, value in spd.items():
        value[0] = (value[0][0], value[0][1], value[0][2], torch.tensor(value[0][3], device=torch_device, dtype=torch.float64, requires_grad=True), value[0][4])

    Mt3dmsClass = Mt3dmsFDM3(x, y, z, total_days, ICBUND, IC, stress_period_step_size, transport_step_size,
                             save_step_size, torch_device, al=al, trpt=trpt, trpv=trpv, prsity=prsity)

    conc = Mt3dmsClass.run(Out, spd)

    # test the result using the Flopy result
    f = h5py.File("./paper_data/output_3.hdf5", "r")
    conc_ground_truth = np.array(f['concentration'])
    heads_ground_truth = np.array(f['head'])
    f.close()

    # visualize the concentration using 2D image
    conc = conc.detach().cpu().numpy()
    conc[conc < 0] = 0
    for i in range(len(conc)):
        visul_tool.simple_plot(conc[i], 'timestep ' + str(i))

    error_percentage1 = np.mean(np.absolute(Out.Phi.detach().cpu().numpy() - heads_ground_truth)) / np.mean(
        heads_ground_truth)
    error_percentage2 = np.mean(np.absolute(conc - conc_ground_truth)) / np.mean(conc_ground_truth)

    print("head error: {}".format(error_percentage1))
    print("concentration error: {}".format(error_percentage2))

    debug = 0




