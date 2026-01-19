import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True #解决中文报错
matplotlib.rc('font', size=12)  #设置字体
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

tau = 5
eta = 0.1
R_0 = 2.14476 #f_0-y
width = 200
eta_0 = eta * width
lambda_min = 1
D = 5420
sigma = 3.5/D



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

def compute_last50_gen_error(train_file, test_file, smooth=1, interval=1):
    train = np.load(train_file, allow_pickle=True)
    test = np.load(test_file, allow_pickle=True)
    data = test - train  # generalization error

    # 平滑
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    # interval 抽样
    data = data[::interval]

    # 取最后50个点
    last50 = data[-10:]
    mean_val = np.mean(last50)
    std_val = np.std(last50)

    print(f"Train File: {train_file}")
    print(f"Test  File: {test_file}")
    print(f"Last 50 Generalization Error Mean : {mean_val:.6f}")
    print(f"Last 50 Generalization Error Std  : {std_val:.6f}")
    print("-" * 60)



fig, ax = plt.subplots(figsize=[5, 4])

#fedavg_0.1
plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
'result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/fedavg5/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
    'result_loss/fedavg5/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy'
)

#fedavg_0.5
plot_learning('result_loss/fedavg4/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg4/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg4/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
'result_loss/fedavg4/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/fedavg4/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg4/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/fedavg4/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
    'result_loss/fedavg4/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy'
)


#fedprox_0.1
plot_learning('result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
    'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy'
)

#fedprox_0.5
plot_learning('result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
    'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/fedavg6/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
    'result_loss/fedavg6/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy'
)

#scaffold_0.1
plot_learning('result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy',
    'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.1resnet18.npy',
    'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.1resnet18.npy'
)

#scaffold_0.5
plot_learning('result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
              ax, label='FedFRTH', linestyle='-', color='tab:red')
plot_learning('result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
              ax, label='FedAvg', linestyle='-', color='tab:green')
compute_last50_gen_error(
    'result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy',
    'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18_freeze.npy'
)
compute_last50_gen_error(
    'result_loss/scaffold/loss_traincifar10_all_data_1_dirichlet_niid_0.5resnet18.npy',
    'result_loss/scaffold/loss_testcifar10_all_data_1_dirichlet_niid_0.5resnet18.npy'
)




# 其他设置
ax.set_xlim([0, 100])
ax.set_ylim([0,1.5])
ax.set_xticks([0,20,40,60,80,100])
ax.set_yticks([0,0.5,1,1.5])
ax.set_xlabel('Global Round')
ax.set_ylabel('Generalization Error(test loss-train loss)')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
# ★ 在 450 处画竖线标记 Stage 2 开始
ax.axvline(x=450, color='black', linestyle=':', linewidth=1.5, label='Stage 2 begins')
ax.legend(handlelength=2.3)



fig.tight_layout()
plt.show()