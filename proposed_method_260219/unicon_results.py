import os
import pandas as pd
import numpy as np
import sys

folder = 'checkpoint_noisy_label'
folder_list = os.listdir(folder)

dataset = 'cifar100'
noise_type = ['sym', 'asym']
sym_list = ['0.2', '0.5', '0.8', '0.9']
asym_list = ['0.1', '0.3', '0.4']
seed_list = [0,1,2]

# Clean
for seed in seed_list:
    exp_dir = os.path.join(folder, f'{dataset}_sym_0.0_{seed}', f'{dataset}_0.0_sym_acc.txt')
    with open(exp_dir, 'r') as f:
        total_acc = f.readlines()
    total_acc = [float(txt) for txt in total_acc]
    total_acc = np.array(total_acc)
    print(exp_dir, len(total_acc), end=' || ')
    print('avg:', round(np.max(total_acc), 2), 'last:', round(np.mean(total_acc[-10:]), 2))
print()

# Sym
for sym_noise in sym_list:
    for seed in seed_list:
        exp_dir = os.path.join(folder, f'{dataset}_sym_{sym_noise}_{seed}', f'{dataset}_{sym_noise}_sym_acc.txt')
        with open(exp_dir, 'r') as f:
            total_acc = f.readlines()
        total_acc = [float(txt) for txt in total_acc]
        total_acc = np.array(total_acc)
        print(exp_dir, len(total_acc), end=' || ')
        print('avg:', round(np.max(total_acc), 2), 'last:', round(np.mean(total_acc[-10:]), 2))
print()

# Asym
for asym_noise in asym_list:
    for seed in seed_list:
        exp_dir = os.path.join(folder, f'{dataset}_asym_{asym_noise}_{seed}', f'{dataset}_{asym_noise}_asym_acc.txt')
        with open(exp_dir, 'r') as f:
            total_acc = f.readlines()
        total_acc = [float(txt) for txt in total_acc]
        total_acc = np.array(total_acc)
        print(exp_dir, len(total_acc), end=' || ')
        print('avg:', round(np.max(total_acc), 2), 'last:', round(np.mean(total_acc[-10:]), 2))