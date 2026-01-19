import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)




def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)



fig, ax = plt.subplots(figsize=[5, 4])

# #0.1
# plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18_freeze_figure4.npy',
#               ax, label='BNAda', linestyle='-', color='tab:red')
# plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.1resnet18_freeze.npy',
#               ax, label='No_BNAda', linestyle='-', color='tab:cyan')


# #0.5
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18_freeze_figure4.npy',
              ax, label='BNAda', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg4/acc_testtinyimagenet_niid_0.5resnet18_freeze.npy',
              ax, label='No_BNAda', linestyle='-', color='tab:cyan')

# 其他设置
ax.set_xlim([0, 100])
ax.set_xticks(np.arange(0,120,20))
# ax.set_ylim([0.45, 0.53])
# ax.set_yticks([0.45,0.47,0.49,0.51,0.53])
ax.set_ylim([0.51, 0.56])
ax.set_yticks([0.51,0.52,0.53,0.54,0.55,0.56])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()

