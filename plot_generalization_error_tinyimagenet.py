import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True #解决中文报错
matplotlib.rc('font', size=12)  #设置字体
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def plot_learning(result_dir_1, result_dir_2, ax, smooth=1, interval=1, **kwargs):
    # 加载一维数据
    data_1 = np.load(result_dir_1,allow_pickle=True)
    data_2 = np.load(result_dir_2, allow_pickle=True)
    data = data_2-data_1

    # 如果 smooth > 1, 可以进行简单的平滑处理，计算滚动平均值
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # 取每 interval 个点
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]

    # 绘制数据
    ax.plot(episode, mean, **kwargs)

def compute_last15_gen_error(train_file, test_file, smooth=1, interval=1):
    train = np.load(train_file, allow_pickle=True)
    test = np.load(test_file, allow_pickle=True)
    data = test - train  # generalization error

    # 平滑
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # interval 抽样
    data = data[::interval]

    # 取最后15个点
    last15 = data[-10:]
    mean_val = np.mean(last15)
    std_val = np.std(last15)

    print(f"Train File: {train_file}")
    print(f"Test  File: {test_file}")
    print(f"Last 15 Generalization Error Mean : {mean_val:.6f}")
    print(f"Last 15 Generalization Error Std  : {std_val:.6f}")
    print("-" * 60)



fig, ax = plt.subplots(figsize=[5, 4])

#fedavg_0.1
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#fedavg_0.5
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg4/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/fedavg4/loss_testtinyimagenet_niid_0.5resnet18.npy'
)
#
#fedprox_0.1
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#fedprox_0.5
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='FedProx', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/fedavg6/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/fedavg6/loss_testtinyimagenet_niid_0.5resnet18.npy'
)

#scaffold_0.1
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18_freeze.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.1resnet18.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.1resnet18.npy'
)

#scaffold_0.5
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18.npy',
'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18.npy',
              ax, label='Scaffold', linestyle='-', color='tab:green')
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18_freeze.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18_freeze.npy'
)
compute_last15_gen_error(
    'result_loss/scaffold/loss_traintinyimagenet_niid_0.5resnet18.npy',
    'result_loss/scaffold/loss_testtinyimagenet_niid_0.5resnet18.npy'
)

# 其他设置
ax.set_xlim([0, 100])
ax.set_ylim([0,4])
ax.set_xticks([0,20,40,60,80,100])
ax.set_yticks([0,1,2,3,4])
ax.set_xlabel('Global Round')
ax.set_ylabel('Generalization Error(test loss-train loss)')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()