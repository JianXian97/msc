# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:49:38 2023

@author: sooji
"""

import matplotlib.pyplot as plt

data = {1: ['SWIN UNETR', 90.8, 394.84],
2: ['UNETR', 76, 41.19],
3: ['UNETR++', 83.28, 31],
4: ['nnUNet ', 83.16, 358],
5: ['CoTr', 84.4, 399.32],
6: ['UNET-MV \n (Proposed)', 80.4, 26],
7: ['nnFormer', 81.6, 213]}

fig, ax = plt.subplots(1,1)

x = [data[i][2] for i in data]
y = [data[i][1] for i in data]
labels = [data[i][0] for i in data]

for i in range(len(data)):
    ax.scatter(x[i],y[i],label=labels[i],s=500)
    if i == 5:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center', weight='bold')
    else:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center')
ax.set_title('Selected model performances on the BTCV dataset, Dice Vs Computational Cost', fontsize=22)
ax.set_ylabel('Dice Score (%)', fontsize=22)
ax.set_xlabel('Computation Cost, FLOPs (G)', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.yaxis.set_tick_params(labelbottom=True)
ax.set_ylim([75,92])
ax.legend(borderpad=2, labelspacing=2)

data = {1: ['SWIN UNETR', 86.37, 668.15],
2: ['UNETR', 78.33, 177.51],
3: ['VNet', 81.96, 849.96],
4: ['UNet ', 150.14, 425.78],
5: ['CoTr', 77.13, 668.15],
6: ['UNET-MV \n (Proposed)', 85.30, 26],
7: ['nnFormer', 85.63, 425.78]}

fig, ax = plt.subplots(1,1)

x = [data[i][2] for i in data]
y = [data[i][1] for i in data]
labels = [data[i][0] for i in data]

for i in range(len(data)):
    ax.scatter(x[i],y[i],label=labels[i],s=500)
    if i == 5:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center', weight='bold')
    else:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center')
ax.set_title('Selected model performances on the AMOS-CT dataset, Dice Vs Computational Cost', fontsize=22)
ax.set_ylabel('Dice Score (%)', fontsize=22)
ax.set_xlabel('Computation Cost, FLOPs (G)', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.yaxis.set_tick_params(labelbottom=True)
ax.set_ylim([75,92])
ax.legend(borderpad=2, labelspacing=2)


data = {1: ['SWIN UNETR', 75.7, 668.15],
2: ['UNETR', 75.3, 177.51],
3: ['VNet', 83.86, 849.96],
4: ['UNet ', 85.59, 425.78],
5: ['CoTr', 77.5, 668.15],
6: ['UNET-MV \n (Proposed)', 80.41, 26],
7: ['nnFormer', 80.6, 425.78]}

fig, ax = plt.subplots(1,1)

x = [data[i][2] for i in data]
y = [data[i][1] for i in data]
labels = [data[i][0] for i in data]

for i in range(len(data)):
    ax.scatter(x[i],y[i],label=labels[i],s=500)
    if i == 5:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center', weight='bold')
    else:
        ax.text(x[i], y[i] + 0.8, labels[i], fontsize=18 ,ha='center', va='center')
ax.set_title('Selected model performances on the AMOS-MRI dataset, Dice Vs Computational Cost', fontsize=22)
ax.set_ylabel('Dice Score (%)', fontsize=22)
ax.set_xlabel('Computation Cost, FLOPs (G)', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.yaxis.set_tick_params(labelbottom=True)
ax.set_ylim([74,92])
ax.legend(borderpad=2, labelspacing=2)