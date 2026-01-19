import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True #解决中文报错
matplotlib.rc('font', size=14)  #设置字体
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

tau = 5
eta = 0.1
R_0 = 2.14476 #f_0-y
width = 200
eta_0 = eta * width
lambda_min = 1
D = 5420
sigma = 3.5/D



def plot_learning(result_dir, ax, smooth=10, interval=1, **kwargs):
    # 加载一维数据
    data = np.load(result_dir,allow_pickle=True)
    data = data[:500]

    # 如果 smooth > 1, 可以进行简单的平滑处理，计算滚动平均值
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # 取每 interval 个点
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]

    # 绘制数据
    ax.plot(episode, mean, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])
#train



#fedavg
plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_random_iidresnet18.npy',
              ax, label='FedAvg_train_iid', linestyle='--', color='tab:cyan')
plot_learning('result_loss/fedavg5/loss_testcifar10_all_data_1_random_iidresnet18.npy',
              ax, label='FedAvg_test_iid', linestyle='--', color='tab:purple')
plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH_train_niid', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH_test_niid', linestyle='-', color='tab:green')





ax.set_xlim([0, 500])
ax.set_ylim([0,3])
ax.set_xticks([0,100,200,300,400,450,500])
ax.set_yticks([0,1,2,3])
ax.set_xlabel('Global Round')
ax.set_ylabel('Loss')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()

# ★ 在 450 处画竖线标记 Stage 2 开始
ax.axvline(x=450, color='black', linestyle=':', linewidth=1.5, label='Stage 2 begins')

ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()