# FedFRTH: Taming Heterogeneity with Fast Convergence and Enhanced Generalization

This repository contains the codes of the paper FedFRTH: Taming Heterogeneity with Fast Convergence and Enhanced Generalization

Our codes are based on the codes for the paper > [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)

## Genarating the IID and non-IID data

```
cd data/cifar10
```
3. Running the `generate_cifar_iid.py` to obtain IID CIFAR-10 data
4. Running the `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of CIFAR-10 with $\alpha=0.1$ and $\alpha=0.5$.

```
cd data/cifar100
```
Running the `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of CIFAR-100 with $\alpha=0.1$ and $\alpha=0.5$.

```
cd data/tinyimagenet
```
## Tiny-ImageNet Dataset Preparation

Due to repository size limitations, the Tiny-ImageNet dataset is not uploaded directly. You need to download and setup the dataset manually.

### 1. Download and Setup
1. Download **Tiny-ImageNet-200** (e.g., from [Stanford CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip)).
2. Navigate to `data/tinyimagenet/` inside this project.
3. **Create a new folder named `data`** inside `tinyimagenet/` if it doesn't exist.
4. Unzip the dataset into that nested `data` folder.

**Correct Directory Structure:**
Ensure your files are organized exactly as shown below. The scripts expect the dataset to be inside a nested `data` folder:

```text
fedavgpy-master/
└── data/
    └── tinyimagenet/
        ├── generate_dirichlet_distribution_0.1_niid.py  <-- Scripts run from here
        ├── generate_dirichlet_distribution_0.5_niid.py
        └── data/                                        <-- Nested 'data' folder
            └── tiny-imagenet-200/                       <-- Dataset goes here
                ├── train/
                ├── val/
                ├── test/
                ├── wnids.txt
                └── words.txt
```

Running the `generate_dirichlet_niid_0.1.py` and `generate_dirichlet_niid_0.5.py` to obtain Dirichlet-based partitions of Tiny-Imagenet with $\alpha=0.1$ and $\alpha=0.5$.

## Note
In our experiment, the number of clients is M=10, and all clients participated in each aggregation process.

If the next experiments using the SGD, you should set

```
cd src/models/client.py
self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
```

If the next experiments using the GD, you should set

```
cd src/models/client.py
self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=Flase) 
self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=Flase)
```

Before running the code, it is necessary to manually set and save the results, for example：running the `main.py` using the `fedavg5` trainer with different networks to obtain figure 1, you must modify the name of the saved file to distinguish among different networks or widths

```
# train the family of fully connected network
np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_fc1', self.loss_list_test)
np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_fc1', self.acc_list_test)
```


## Impact of Non-IID Versus Network Width

Running the `main.py` using the `fedavg5` trainer with different networks to obtain figure 1 and figure 2.

Running the `main.py` using the `fedavg4` trainer with different networks to obtain the variation of global NTK and parameters in figure 3 and running the `main.py` using the `fedavg12` trainer with different networks to obtain the variation of local NTK in figure 3.

## Linear Approximation of FedAvg

Running the `main.py` using the `fedavg9` trainer with the fully-connected networks to obtain figure 4.

## FedAvg Evolves as Centralized Learning

Running the `main.py` using the `fedavg11` trainer with the fully-connected networks to obtain figure 6.


## Dependency

python = 3.8.18

pytorch = 1.9.1

CUDA = 11.1

Tensordboardx = 2.6.2.2

Numpy = 1.24.3


