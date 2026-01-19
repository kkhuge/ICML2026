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



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    # 加载一维数据
    data = np.load(result_dir,allow_pickle=True)

    # if result_dir == 'result_acc/fedavg5/acc_testcifar100_all_data_1_dirichlet_niidresnet18_freeze.npy':
    #     data = data[-99:]

    # 如果 smooth > 1, 可以进行简单的平滑处理，计算滚动平均值
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # 取每 interval 个点
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]

    # 绘制数据
    ax.plot(episode, mean, **kwargs)

# ====== 计算最后 50 轮均值和方差 ======
def compute_last50_stats(result_dir):
    data = np.load(result_dir, allow_pickle=True)

    # 如果想保持和绘图一致（用了 smooth 或 interval），也可以做相同处理
    # 这里按原始数据计算最后50个点
    last50 = data[-50:]
    mean_val = np.mean(last50)
    std_val = np.std(last50)

    print(f"File: {result_dir}")
    print(f"Last 50 rounds mean: {mean_val:.6f}")
    print(f"Last 50 rounds std: {std_val:.6f}")
    print("-" * 60)



fig, ax = plt.subplots(figsize=[5, 4])
#train



# # #0.1
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau1.npy',
#               ax, label=r'$\tau=1$', linestyle='-', color='tab:cyan')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau5.npy',
#               ax, label=r'$\tau=5$', linestyle='-', color='tab:red')
# # plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau10.npy',
# #               ax, label=r'$\tau=10$', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.1resnet18_freeze_tau50.npy',
#               ax, label=r'$\tau=50$', linestyle='-', color='tab:purple')
#
#
# #
# # 其他设置
# ax.set_xlim([0, 100])
# ax.set_ylim([0.55, 0.65])
# ax.set_xticks(np.arange(0,120,20))
# ax.set_yticks([0.55,0.57,0.59,0.61,0.63,0.65])
# ax.set_xlabel('Global Round')
# ax.set_ylabel('Test Accuracy')
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.grid()
# ax.legend(handlelength=2.3)
# fig.tight_layout()
# plt.show()




# #0.5
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau1.npy',
              ax, label=r'$\tau=1$', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau5.npy',
              ax, label=r'$\tau=5$', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau10.npy',
#               ax, label=r'$\tau=10$', linestyle='-', color='tab:green')
plot_learning('result_acc/fedavg4/acc_testcifar100_all_data_1_dirichlet_niid_0.5resnet18_freeze_tau50.npy',
              ax, label=r'$\tau=50$', linestyle='-', color='tab:purple')


#
# 其他设置
ax.set_xlim([0, 100])
ax.set_ylim([0.62, 0.68])
ax.set_xticks(np.arange(0,120,20))
ax.set_yticks([0.62,0.64,0.66,0.68])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()

