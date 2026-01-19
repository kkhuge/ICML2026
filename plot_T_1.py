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
    # 如果 smooth > 1, 可以进行简单的平滑处理，计算滚动平均值
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # 取每 interval 个点
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]

    # 绘制数据
    ax.plot(episode, mean, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])

#
# # cifar100_0.1
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_50.npy',
#               ax, label='$T_1=50$', linestyle='-', color='tab:orange')
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_100.npy',
#               ax, label='$T_1=100$', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_200.npy',
#               ax, label='$T_1=200$', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_400.npy',
#               ax, label='$T_1=400$', linestyle='-', color='tab:purple')
# plot_learning('result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_600.npy',
#               ax, label='FedAvg', linestyle='-', color='tab:cyan')
# #
#
#
# # 其他设置
# ax.set_xlim([0, 600])
# ax.set_ylim([0, 0.7])
# ax.set_xticks(np.arange(0,700,100))
# ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
# ax.set_xlabel('Global Round')
# ax.set_ylabel('Test Accuracy')
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.grid()
# ax.legend(handlelength=2.3)
# fig.tight_layout()
# plt.show()



# tiny_0.1
plot_learning('result_acc/fedavg5/acc_testtinyimagenet_niid_0.1resnet18_freeze_50.npy',
              ax, label='$T_1=50$', linestyle='-', color='tab:orange')
plot_learning('result_acc/fedavg5/acc_testtinyimagenet_niid_0.1resnet18_freeze_100.npy',
              ax, label='$T_1=100$', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedavg5/acc_testtinyimagenet_niid_0.1resnet18_freeze_200.npy',
#               ax, label='$T_1=200$', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testtinyimagenet_niid_0.1resnet18_freeze_400.npy',
#               ax, label='$T_1=400$', linestyle='-', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testtinyimagenet_niid_0.1resnet18_freeze_600.npy',
              ax, label='FedAvg', linestyle='-', color='tab:cyan')
#


# 其他设置
ax.set_xlim([0, 600])
ax.set_ylim([0, 0.7])
ax.set_xticks(np.arange(0,700,100))
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()
