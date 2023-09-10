# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:38:19 2023

@author: sooji
"""

import matplotlib.pyplot as plt
import pandas as pd

data = {1: ['SWIN UNETR', 90.8, 394.84],
2: ['UNETR', 76, 41.19],
3: ['UNETR++', 83.28, 31],
4: ['nnUNet ', 83.16, 358],
5: ['CoTr', 84.4, 399.32],
6: ['UNET-MV \n (Mine)', 81.0, 26],
7: ['nnFormer', 81.6, 150.5]}



x = [data[i][2] for i in data]
y = [data[i][1] for i in data]
labels = [data[i][0] for i in data]


df = pd.read_pickle(r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\Tune Expt Results DOC.pkl")
df.reset_index(inplace = True)

ranges = {
    'E' : (None,4),
    'F' : (5,9),
    'cft_mode': (10,12),
    'decode_mode':(13,15)
    }

labels = {
    'E' : "Transformer Dimension, E",
    'F' : "Model Dimension, F",
    'cft_mode' : "CFT model configuration",
    'decode_mode' : "decoder model configuration"
    }

def plot(var):
    m,n = ranges[var]
    
    fig, ax1 = plt.subplots(1,1)
    ax1color = 'tab:red'
    lns1 = ax1.plot(df.loc[m:n, var], df.loc[m:n, 'dice_avg'] * 100, color=ax1color, label = 'Dice Score', marker='o')
    # ax1.set_ylim([75,90])
    
    ax2 = ax1.twinx() 
    ax2color = 'tab:blue'
    lns2 = ax2.plot(df.loc[m:n, var], df.loc[m:n, 'hd_avg'], color=ax2color, label = '95HD', marker='o')
    # ax2.set_ylim([4,6])
    
    ax1.set_title('Dice and 95HD scores with respect to varying ' + labels[var], fontsize=22)
    ax1.set_ylabel('Dice Score (%)', fontsize=22)
    ax2.set_ylabel('95HD (mm)', fontsize=22)
    ax1.set_xlabel(labels[var], fontsize=22)
    ax1.tick_params(axis='y', which='major', labelsize=22)
    ax1.tick_params(axis='x', which='major', labelsize=22)
    ax2.tick_params(axis='y', which='major', labelsize=22)
    ax1.yaxis.set_tick_params(labelbottom=True)
    
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0,  fontsize="20")

plot('F')