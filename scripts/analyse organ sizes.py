# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:42:48 2023

@author: sooji
"""

import json

import nibabel as nib 
import numpy as np
from os import listdir
from os.path import isfile, join
from si_prefix import si_format

def print_nice(mean, std):
    output1 = ""
    output2 = ""
    output3 = ""
    for i, (mean, std) in enumerate(zip(list(mean.values())[1:], list(std.values())[1:])):   
        mean = si_format(mean, 0)
        std = si_format(std, 0)
        if i < 5:
            output1 += (mean + " ± " + std)
            output1 += " & "
        elif i < 10:
            output2 += (mean + " ± " + std)
            output2 += " & "
        else:
            output3 += (mean + " ± " + std)
            output3 += " & "
    
    print(output1)
    print(output2)
    print(output3)
    
def calc_ratio(mean, std):
    z = [y/x for x,y in zip(mean.values(), std.values())]
    
    output1 = "& "
    output2 = "& "
    for i in range(1, len(z)):
        if i < 8:
            output1 += "{:.2f}".format(z[i])
            output1 += " & "
        elif i == len(z)-1:
            output2 += "{:.2f}".format(z[i])
            output2 += " \\"
        else:
            output2 += "{:.2f}".format(z[i])
            output2 += " & " 
    print(output1)
    print(output2)

BTCV_path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\BTCV\labelsTr"
AMOS_path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\AMOS\labelsTr"
BTCV_files = [join(BTCV_path, f) for f in listdir(BTCV_path) if isfile(join(BTCV_path, f))]
AMOS_files = [join(AMOS_path, f) for f in listdir(AMOS_path) if isfile(join(AMOS_path, f))]

BTCV_organs = 13
AMOS_organs = 15

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

AMOS_MRI_path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\AMOS\dataset_MRI.json"

with open(AMOS_MRI_path) as f:
    d = json.load(f)['training']
    AMOS_MRI_set = set(f['image'].split('/')[-1] for f in d)
    
    
BTCV = {i:[] for i in range(BTCV_organs+1)}
BTCV_percent = {i:[] for i in range(BTCV_organs+1)}
AMOS_CT = {i:[] for i in range(AMOS_organs+1)}
AMOS_MRI = {i:[] for i in range(AMOS_organs+1)}
AMOS_CT_percent = {i:[] for i in range(AMOS_organs+1)}
AMOS_MRI_percent = {i:[] for i in range(AMOS_organs+1)}
for path in BTCV_files:
    seg_map = nib.load(path).get_fdata().astype(int)
    total = (seg_map > 0).sum()
    for i in range(1,BTCV_organs+1):
        BTCV_percent[i].append((seg_map == i).sum()/total*100)
        BTCV[i].append((seg_map == i).sum())


from si_prefix import si_format
BTCV_mean = {newkey: np.mean(BTCV[oldkey]) for (oldkey, newkey) in BTCV_map.items()}
BTCV_std = {newkey: np.std(BTCV[oldkey]) for (oldkey, newkey) in BTCV_map.items()}
BTCV_std_percent = [y/x for x,y in zip(BTCV_mean.values(), BTCV_std.values())]

print_nice(BTCV_mean, BTCV_std)


for j, path in enumerate(AMOS_files):
    print("AMOS " + str(j))
    seg_map = nib.load(path).get_fdata().astype(int)
    total = (seg_map > 0).sum()
    name = path.split("\\")[-1]
    isMRI = name in AMOS_MRI_set
    for i in range(1,AMOS_organs+1):
        if isMRI:
            AMOS_MRI_percent[i].append((seg_map == i).sum()/total*100)
            AMOS_MRI[i].append((seg_map == i).sum())
        else:
            AMOS_CT_percent[i].append((seg_map == i).sum()/total*100)
            AMOS_CT[i].append((seg_map == i).sum())
    
       
        
AMOS_CT_mean = {newkey: np.mean(AMOS_CT[oldkey]) for (oldkey, newkey) in AMOS_map.items()}
AMOS_CT_std = {newkey: np.std(AMOS_CT[oldkey]) for (oldkey, newkey) in AMOS_map.items()}
AMOS_MRI_mean = {newkey: np.mean(AMOS_MRI[oldkey]) for (oldkey, newkey) in AMOS_map.items()}
AMOS_MRI_std = {newkey: np.std(AMOS_MRI[oldkey]) for (oldkey, newkey) in AMOS_map.items()}
output1 = ""
output2 = ""

print_nice(AMOS_CT_mean, AMOS_CT_std)
print_nice(AMOS_MRI_mean, AMOS_MRI_std)


import pickle
data = {}
data['AMOS_CT_mean'] = AMOS_CT_mean
data['AMOS_CT_std'] = AMOS_CT_std
data['AMOS_CT_percent'] = AMOS_CT_percent
data['AMOS_MRI_mean'] = AMOS_MRI_mean
data['AMOS_MRI_std'] = AMOS_MRI_std
data['AMOS_MRI_percent'] = AMOS_MRI_percent
data['BTCV_mean'] = BTCV_mean
data['BTCV_std'] = BTCV_std
data['BTCV_percent'] = BTCV_percent

file = open(r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\data_set_analysis.pkl","wb")
pickle.dump(data, file)