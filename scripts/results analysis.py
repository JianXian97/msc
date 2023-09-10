# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 20:36:22 2023

@author: sooji
"""
import scipy
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import colors


cmap = plt.cm.jet   
cmaplist = [cmap(i) for i in range(cmap.N)][::-1]
cmaplist[0] = (0, 0, 0, 1)
cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
 
def plot(a, b, axis_x, axis_y, label_x, label_y, outliers, label_map):
    a_clean, b_clean = a.copy(), b.copy()
    
    color_step = len(cmaplist) // len(label_map)
    
    for pos in outliers[::-1]:
        del a_clean[pos]
        del b_clean[pos]
    
    print(pearsonr(a_clean,b_clean))
    
    fig, ax1 = plt.subplots(1,1)
    for i in range(len(a)):
        ax1.scatter(a[i],b[i],label=label_map[i+1], s= 200, color = cmaplist[i * color_step])
    
    x = list(np.linspace(min(a_clean), max(a_clean)))
    ax1.plot(x, np.poly1d(np.polyfit(a_clean, b_clean, 1))(x))
    ax1.set_title('UNET-MV ' + axis_y + ' vs ' + axis_x, fontsize=22)
    ax1.set_ylabel(label_y, fontsize=22)
    ax1.set_xlabel(label_x, fontsize=22)
    ax1.tick_params(axis='y', which='major', labelsize=22)
    ax1.tick_params(axis='x', which='major', labelsize=22)
    ax1.legend(fontsize="14")
    
BTCV_map = {
0: 'Clear Label',
1: 'Spleen',
2: 'Right Kidney',
3: 'Left Kidney',
4: 'Gall Bladder',
5: 'Esophagus',
6: 'Liver',
7: 'Stomach',
8: 'Aorta',
9: 'Inferior Vena Cava',
10: 'Portal and Splenic Veins',
11: 'Pancreas',
12: 'Right Adrenal Gland',
13: 'Left Adrenal Gland'}    
 

hd = [1.44, 1.74, 1.79, 25.4, 4.67, 2.19, 13.79, 7.32, 5.00, 6.03, 4.23, 2.72, 4.68]
coeff_var = [1.46, 0.42, 0.44, 0.87, 0.50, 0.31, 0.57, 0.50, 0.44, 0.59, 0.37, 0.36, 0.39]

plot(coeff_var, hd, "BTCV Organ coefficient of variation", "BTCV 95HD scores",  'BTCV Organ coefficient of variation','95HD (mm)', [0], BTCV_map)


p = [9.1, 5.1, 5.0, 0.8, 0.5, 55.5, 13.8, 3.2, 2.8, 1.1, 2.8, 0.1, 0.2]
dice = [94.8, 93.2, 94.8, 58.6, 75.1, 96.6, 83.3, 89.1, 82.1, 72.7, 79.3, 65.2, 60.1]
plot(p,dice, "BTCV Organ relative sizes", "BTCV Dice", "Organ relative sizes (%)", "Dice (%)", [5,6], BTCV_map)


AMOS_map={
0: 'Clear Label',
1: 'Spleen',
2: 'Right Kidney',
3: 'Left Kidney',
4: 'Gall Bladder',
5: 'Esophagus',
6: 'Liver',
7: 'Stomach',
8: 'Aorta',
9: 'Inferior Vena Cava',
10: 'Pancreas',
11: 'Right Adrenal Gland',
12: 'Left Adrenal Gland',
13: 'Duodenum',
14: 'Bladder',
15: 'Prostate/Uterus'}

hd = [2.9, 6.58, 5.17, 4.36, 3.05, 5.2, 8.11, 2.67, 2.3, 3.88, 3.89, 3.49, 5.81, 6.45, 4.77]
coeff_var = [0.78 ,0.74 ,0.82 ,1.10 ,0.93 ,0.65 ,1.09 ,0.95 ,0.63 ,0.69 ,0.72 ,0.71 ,0.76 ,1.71 ,1.35]

plot(coeff_var, hd,"AMOS-CT Organ coefficient of variation", "AMOS-CT 95HD scores", 'AMOS-CT Organ coefficient of variation', '95HD (mm)', [1,6,12], AMOS_map)


p = [7.34 ,5.44 ,5.72 ,1.09 ,0.59 ,49.07 ,11.94 ,4.52  ,2.67 ,2.69 ,0.13 ,0.15 ,2.13 ,4.55 ,1.97]
dice = [95.21, 93.55, 95.11, 77.93, 79.6, 96.96, 89.26, 93.79, 88.9, 83.58, 72.94, 69.42, 77.8, 86.22, 79.24]
plot(p,dice, "AMOS-CT Organ relative sizes", "AMOS-CT Dice", "Organ relative sizes (%)", "Dice (%)", [5,6], AMOS_map)

MRI_map={
0: 'Clear Label',
1: 'Spleen',
2: 'Right Kidney',
3: 'Left Kidney',
4: 'Gall Bladder',
5: 'Esophagus',
6: 'Liver',
7: 'Stomach',
8: 'Aorta',
9: 'Inferior Vena Cava',
10: 'Pancreas',
11: 'Right Adrenal Gland',
12: 'Left Adrenal Gland',
13: 'Duodenum',
}

hd = [5.43, 5.82, 3.18, 6.86, 3.53, 2.32, 4.6, 4.69, 2.15, 3.68, 3.59, 4.58, 7.96]
coeff_var = [1.17 , 0.87 , 0.95 , 1.38 , 1.01 , 0.93 , 0.97 , 1.32 , 0.88 , 1.03 , 0.97 , 1.10 , 0.88]

plot(coeff_var, hd,"AMOS-MRI Organ coefficient of variation", "AMOS-MRI 95HD scores", 'AMOS-MRI Organ coefficient of variation', '95HD (mm)', [1,12], MRI_map)


p = [9.31 , 7.02 , 6.72 , 1.28 , 0.41 , 54.97 , 7.89 , 4.15 , 2.35 , 3.64 , 0.10 , 0.13 , 1.92]
dice = [95.33, 95.24, 94.2, 71.99, 70.16, 97.13, 88.31, 91.47, 86.09, 81.45, 57.87, 53.91, 61.81]
plot(p,dice, "AMOS-MRI Organ relative sizes", "AMOS-MRI Dice", "Organ relative sizes (%)", "Dice (%)", [5], AMOS_map)
