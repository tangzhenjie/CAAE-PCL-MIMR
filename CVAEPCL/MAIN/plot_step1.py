import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import matplotlib.ticker as ticker
import os

mpl.rcParams['figure.figsize'] = (8, 6)
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

def softmax(x):
    """Compute softmax values for scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

inversion_step1_dir = "./inversion_step1/idx_1946/"
save_dir = "./step1_plot_result/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if __name__ == '__main__':
    iter = 80
    probabilities = []
    zeros = np.zeros(12)
    init_probability = softmax(zeros)
    probabilities.append(init_probability)
    for i in range(1, iter+1):
        file_name = "inversion_step1_result" + str(i) + ".hdf5"
        f_out = h5py.File(inversion_step1_dir + file_name, "r")
        probability = np.array(f_out['probability'])
        f_out.close()
        probabilities.append(probability)

    x = np.linspace(0, iter, iter + 1)
    ys = np.array(probabilities)

    # curves
    for i in range(12):
        if i == 10:
            y = ys[:, i]
            label = r'$w_{{{}}}$'.format(i+1)
            plt.plot(x, y, label=label, linestyle='--')
        else:
            y = ys[:, i]
            label = r'$w_{{{}}}$'.format(i+1)
            plt.plot(x, y, label=label)

    #plt.title('Possibilities for All Source Positions')

    plt.xlabel('Iterations')
    plt.ylabel('Probability')

    plt.grid(True)  # 添加网格线
    plt.xlim(0, 80)  # 设置 x 轴范围从 0 到 80
    x_ticks = np.linspace(0, 80, 9)
    plt.xticks(x_ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'location_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


    """
            可视化损失函数
    """
    f_out = h5py.File(inversion_step1_dir + "loss_result.hdf5", "r")
    loss_result = np.array(f_out['loss_result'])
    f_out.close()

    Iterations = [i + 1 for i in range(len(loss_result[0:-1]))]
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.plot(Iterations, loss_result[0:-1], 'k', label='Loss values')
    # axs.set_title('Loss Values')
    axs.set_xlabel('Iterations')
    # axs.set_ylabel('Loss')
    axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    axs.grid(True)  # 添加网格线
    plt.xlim(-1, 80)  # 设置 x 轴范围从 0 到 80
    x_ticks = np.linspace(0, 80, 9)
    plt.xticks(x_ticks)
    # axs.legend()
    plt.tight_layout()
    plt.savefig(save_dir + 'loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
