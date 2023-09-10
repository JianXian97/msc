# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:08:14 2223

@author: sooji
"""

import matplotlib.pyplot as plt
import itertools
hyper_params = {
    'mode': ['CA', 'simple'],
    'cct_mode': ['channel', 'patch', 'all']
}
combinations = list(itertools.product(*hyper_params.values()))

# path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\train_BTCV_mv_cft.out"
path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\train_BTCV_mv_cft2_tune_archi.out"

data = {}
count = 0

with open(path, 'r', encoding="utf-8") as f:
    for line in f:
        if "Final training" in line:
            seq = int(line.split("/")[0].split(" ")[-1])
            if seq == 0:
                count += 1
                data[count] = []
            loss = float(line.split("loss: ")[-1].split(" time")[0])
            data[count].append(loss)
        
fig, ax = plt.subplots(2,3,sharey='row')
plt.subplots_adjust(top = 0.90, bottom=0.1, left=0.04, right=0.96, hspace=0.4, wspace=0.2)
for i in range(6):
    mode, cft_mode = combinations[i][0], combinations[i][1]
    losses = data[i+1][:5000]
    x,y = int(i/3), i%3
    ax[x,y].plot(losses)
    ax[x,y].set_title('Config: ' + mode + ', ' + cft_mode, fontsize=22)
    ax[x,y].set_ylabel('Loss', fontsize=22)
    ax[x,y].set_xlabel('Epochs', fontsize=22)
    ax[x,y].tick_params(axis='both', which='major', labelsize=22)
    ax[x,y].yaxis.set_tick_params(labelbottom=True)