import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 6)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
# plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
# Set the default serif font to Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'  # Ensure that the serif font is used
plt.rcParams['axes.linewidth'] = 2.0  # 边界框
#plt.rcParams['text.usetex'] = True
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# 生成数据
x_values = np.linspace(0, 2500, 82)  # 生成 x 轴的边界值
y_values = np.linspace(0, 1250, 42)  # 生成 y 轴的边界值

"""
    plot candidate  observation wells, as background
"""

y_obs_data = np.linspace(1, 39, num=20).astype('int')
x_obs_data = np.linspace(1, 79, num=40).astype('int')
y_obs = []
x_obs = []
for y in y_obs_data:
    for x in x_obs_data:
        x_obs.append(x)
        y_obs.append(y)

# # 绘制散点图
# for i in range(len(x_values) - 1):
#     x_center = (x_values[i] + x_values[i+1]) / 2  # 计算 x 轴中心值
#     for j in range(len(y_values) - 1):
#         y_center = (y_values[j] + y_values[j+1]) / 2  # 计算 y 轴中心值
#         plt.scatter(x_center, y_center, color='black', alpha=0.7, s=5)  # 在每个网格的中心位置绘制散点图

# 创建一个新的 Figure 对象
fig, ax = plt.subplots()

# 绘制candidate observation wells
for i in range(len(y_obs)):
    y_index = int(y_obs[i])
    x_index = int(x_obs[i])
    x_center = (x_values[x_index] + x_values[x_index + 1]) / 2  # 计算 x 轴中心值
    y_center = (y_values[y_index] + y_values[y_index + 1]) / 2  # 计算 y 轴中心值
    ax.scatter(x_center, y_center, color='b', marker='o', alpha=0.9, s=5)  # 绘制强调点
    # plt.text(x_center, y_center, f'({x_index}, {y_index})', fontsize=8, ha='left', va='bottom')  # 添加索引位置

"""
    plot the selected wells
"""
input_file = "./MIMR_results/MIMR_head_Pct0.9.hdf5"
f = h5py.File(input_file, "r")
y_obs_select = np.array(f["y_obs"])
x_obs_select = np.array(f["x_obs"])
f.close()


for i in range(len(y_obs_select)):
    y_index = int(y_obs_select[i])
    x_index = int(x_obs_select[i])
    x_center = (x_values[x_index] + x_values[x_index + 1]) / 2  # 计算 x 轴中心值
    y_center = (y_values[y_index] + y_values[y_index + 1]) / 2  # 计算 y 轴中心值
    plt.scatter(x_center, y_center, color='red', marker='*', alpha=0.9, s=100)  # 绘制强调点
    # plt.text(x_center, y_center, f'({x_index}, {y_index})', fontsize=8, ha='left', va='bottom')  # 添加索引位置


# 设置 x 和 y 轴的取值范围
ax.set_xlim(0, 2500)
ax.set_ylim(0, 1250)
ax.set_xticks(np.linspace(0, 2500, 6))
ax.set_yticks(np.linspace(0, 1250, 6))  # 设置 y 轴刻度为 0 到 1250，共 6 个刻度

# 显示网格
ax.grid(False)

save_dir = "./plot_select_wells/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# axs.legend()
plt.tight_layout()
plt.show()
fig.savefig(save_dir + input_file[15:-5] + '.png', dpi=300, bbox_inches='tight')

