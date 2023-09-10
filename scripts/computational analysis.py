# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:55:26 2023

@author: sooji
"""
from pytexit import py2tex
#py2tex('eqn')

import numpy as np
import itertools
global incl_attn 

def nicePrint(module, component_names, *values):
    pass
    # assert len(component_names) == len(values), "Number of component names and values don't match"
    # values = np.array(values)
    # percentage = values / values.sum() * 100
    
    # print("_"*26)
    # print(module)
    # for i in range(len(percentage)):
    #     name, p = component_names[i], percentage[i]
    #     string = "Name: {0:15} {1:.2f}".format(name, p)
    #     print(string)
        
def mobileVit(P, E, E_mlp, I, Cin, F):
    local = max(max(P)//2, 1) * (27 * Cin**2 * I.prod() + 2 * Cin * I.prod()) + I.prod() * E * Cin + 2 * E * I.prod()
    global_ = 2 * I.prod() * E**2 + 2 * I.prod() * E + 2 * E ** 2 * (I/P).prod() + 3 * (2 * E_mlp + 8) * (I/P).prod() * E**2 + 12 * (I/P).prod() * E**3 + 6 * (I[0]/P[0])**4 * E**2
    if not incl_attn:
        global_ = 2 * I.prod() * E**2 + 2 * I.prod() * E + 2 * E ** 2 * (I/P).prod() + 3 * (2 * E_mlp + 8) * (I/P).prod() * E**2 + 12 * (I/P).prod() * E**3 
    global_fold_unfold = 2 * I.prod() * E**2 + 2 * I.prod() * E + 2 * E ** 2 * (I/P).prod()
    global_trans =  3 * (2 * E_mlp + 8) * (I/P).prod() * E**2 + 12 * (I/P).prod() * E**3 + 6 * (I[0]/P[0])**4 * E**2
    nicePrint("MobileVit - GLOBAL", ['global_fold_unfold', 'global_trans'], global_fold_unfold, global_trans)
    
    fusion = Cin * I.prod() * E + 2 * Cin * I.prod() + (2 * Cin + E) * F * I.prod() + 2 * F * I.prod()
    
    total = local + global_ + fusion
    nicePrint("MobileVit", ['local', 'global', 'fusion'], local, global_, fusion)
    return total

def downsample(I, F):
    return 3 * 27 * F**2 * (I/2).prod() + 4 * F * (I/2).prod()

def encoder(P, E, E_mlp, I, F):
    mv = 0
    ds = 0
    for i in range(5):
        P_ = P / 2**i
        I_ = I / 2**i
        Cin_ = F / 2 if i == 0 else F * 2**(i-1)
        F_ = F * 2**i
        mv += mobileVit(P_, E, E_mlp, I_, Cin_, F_)
        
        if i == 4:
            break
        ds += downsample(I_, F_)
    return mv + ds

def upsampling(P, E, E_mlp, I, F, mode="CA"):
    trans_conv = 16 * I.prod() * F**2
    trans_proj = 2 * (I.prod() * E * F + 2 * I.prod() * E)
    unfold_proj = 2 * (I.prod() * E**2 + 2 * E**2 * (I/P).prod())
    CA_trans = 2 * E**2 * E_mlp * (I/P).prod() + 12 * E**2 * (I/P).prod() + 4 * E**3 * (I/P).prod() + 2 * E**2 * ((I/P).prod())**2
    if not incl_attn:
        CA_trans = 2 * E**2 * E_mlp * (I/P).prod() + 12 * E**2 * (I/P).prod() + 4 * E**3 * (I/P).prod()  
    fold_proj = I.prod() * E**2 + 2 * I.prod() * E
    
    
    if mode == "CA":
        conv_block = 2 * E * I.prod() * F + I.prod() * F**2 + 6 * I.prod() * F
        total = trans_conv + trans_proj + unfold_proj + CA_trans + fold_proj + conv_block
        nicePrint("CA Upsampling", ['trans_conv', 'trans_proj', 'unfold_proj', 'CA_trans', 'fold_proj', 'conv_block'], trans_conv, trans_proj, unfold_proj, CA_trans, fold_proj, conv_block)
    else:
        conv_block = 5 * I.prod() * F**2 + 6 * I.prod() * F
        total = trans_conv + conv_block
        nicePrint("Simple Upsampling", ['trans_conv', 'conv_block'], trans_conv, conv_block)

    return total

def decoder(P, E, E_mlp, I, F, mode="CA"):
    total = 0
    for i in range(4):
        P_ = P / 2**i
        I_ = I / 2**i
        F_ = F * 2**i
        total += upsampling(P_, E, E_mlp, I_, F_, mode)
    
    # print(total)
    return total

def cft(P, E, E_mlp, I, F, mode="all"):
    if mode == "skip":
        return 0
    patch_attn = 0
    
    global_proj = (85 / 32) * I.prod() * F * (E + 1) + (585 / 256) * I.prod() * E
    unfold_proj = (585 / 512) * I.prod() * E**2 + 8 * E**2 * (I/P).prod()
    fold_proj = (585 / 512) * I.prod() * E**2 + (585 / 256) * I.prod() * E
    patch_trans = 4 * (2 * E**2 * (I/P).prod() * E_mlp + 24 * E**2 * (I/P).prod() + 10 * E**3 * (I/P).prod() + 8 * E**2 * ((I/P).prod())**2)
    if not incl_attn:
        patch_trans = 4 * (2 * E**2 * (I/P).prod() * E_mlp + 24 * E**2 * (I/P).prod() + 10 * E**3 * (I/P).prod())
    patch_attn = global_proj + unfold_proj + fold_proj + patch_trans
    
    if mode != "channel":
        nicePrint("PATCH TRANSFORMER", ['global_proj', 'unfold_proj', 'fold_proj', 'patch_trans'], global_proj, unfold_proj, fold_proj, patch_trans)
    
    channel_attn = 0 
    
    unfold_proj = (85 / 64) * I.prod() * E * F + 30 * (I/P).prod() * E * F 
    fold_proj = (85 / 64) * I.prod() * E * F + (85 / 32) * I.prod() * F
    channel_trans = 270 * E * F * ((I/P).prod())**2 + 360 * E * F * (I/P).prod() + 450 * E * F**2 * (I/P).prod()
    if not incl_attn:
        channel_trans = 270 * E * F * ((I/P).prod())**2 + 360 * E * F * (I/P).prod() 
    channel_attn += unfold_proj + fold_proj + channel_trans
    
    if mode != "channel":
        nicePrint("CHANNEL TRANSFORMER", ['unfold_proj', 'fold_proj', 'channel_trans'], unfold_proj, fold_proj, channel_trans)
    

    fusion = (15 / 4) * I.prod() * F**2 + (85 / 32) * F * I.prod()
    
    if mode == "all":
        total = patch_attn + channel_attn + fusion
        nicePrint("CFT ALL", ['patch_attn', 'channel_attn', 'fusion'], patch_attn, channel_attn, fusion)
        
    elif mode == "patch":
        total = patch_attn + fusion
        nicePrint("CFT ALL", ['patch_attn', 'fusion'], patch_attn, fusion)
    
    elif mode == "channel":
        total = channel_attn + fusion
        nicePrint("CFT ALL", ['channel_attn', 'fusion'], channel_attn, fusion)
    
    return total
    
def preprocess(C, I, F):
    return 2 * 27 * C * F * (I/2).prod() + 1 * 27 * F**2 * (I/2).prod() + 4 * F * (I/2).prod()

def skip(C, I, F):
    return 28 * C * F * I.prod() + 1 * 27 * F**2 * I.prod() + 4 * F * I.prod() 

# def postprocess(P, E, E_mlp, I, F, mode):
#     return upsampling(P*2, E, E_mlp, I, F, mode)

def postprocess(I, F):
    return 2 * I.prod() * F**2 * 8 

def out(Cout, I,F):
    return I.prod() * 1 * F * Cout

def model(C, Cout, I, F, P, E, mode, cft_mode):
    #add preprocess / post process
    E_mlp = E * 4
    extra = preprocess(C, I, F) + postprocess(I, F) 
    # extra = preprocess(C, I, F) + postprocess(P, E, E_mlp, I, F, mode) + skip(C, I, F)
    outval = out(Cout, I, F)

    I //= 2
    F *= 2

    enc = encoder(P, E, E_mlp, I, F)
    dec = decoder(P, E, E_mlp, I, F, mode)
    cft_val = cft(P, E, E_mlp, I, F, cft_mode)
    
    # '''
    # print(extra)
    # print(enc)
    # print(dec)
    # print(cft_val)
    # print(outval)
    print(extra + outval + enc + dec + cft_val)
    # '''
    
    return extra + outval + enc + dec + cft_val



import matplotlib.pyplot as plt
def plot_cost(C, Cout, I, F, P, E, mode, cft_mode):
    num_pts = 5
    params = {'E' : [E for i in range(num_pts)],
              'F' : [F for i in range(num_pts)]
        }
    points = {'E' : [18,36,54,72,90],
              'F' : [4,8,12,16,20]
        }
    
    fig, ax = plt.subplots(1,2,sharey='row')
    
    for pos, var in enumerate(['E', 'F']):
        costs = []
        new_params = params.copy()
        new_params[var] = points[var]
        for i in range(num_pts):
            cost = model(C, Cout, I, new_params['F'][i], P, new_params['E'][i], mode, cft_mode)/10**9
            costs.append(cost)
            label = "(" + str(points[var][i]) + ", " + str(int(cost)) + ")"
            ax[pos].text(points[var][i], cost  + (-1)**i * 1, label, fontsize=18 ,ha='center', va='center')
             
        ax[pos].plot(points[var], costs)
        ax[pos].scatter(points[var], costs)
        ax[pos].set_title('Cost with respect to changes in ' + var, fontsize=22)
        ax[pos].set_ylabel('Multiply Accumulate Operations (G)', fontsize=22)
        ax[pos].set_xlabel('Values taken by parameter ' + var, fontsize=22)
        ax[pos].tick_params(axis='both', which='major', labelsize=22)
        ax[pos].yaxis.set_tick_params(labelbottom=True)
        

    
#mobile vit
C = 1
P = np.array([16,16,16], dtype=np.int64)
E = 72
I = np.array([96,96,96], dtype=np.int64)
F = 16
Cout = 16
mode = "simple"
cft_mode = "channel"

incl_attn = True

hyper_params = {
    'mode': ['CA', 'simple'],
    'cft_mode': ['channel', 'patch', 'all', 'skip']
}
combinations = list(itertools.product(*hyper_params.values()))
model(C, Cout, I.copy(), F, P.copy(), E, mode, cft_mode)
for c in combinations:
    mode = c[0]
    cft_mode = c[1]
    print(mode + " " + cft_mode, end=" ")
    model(C, Cout, I.copy(), F, P.copy(), E, mode, cft_mode)

plot_cost(C, Cout, I.copy(), F, P.copy(), E, mode, cft_mode)

# '''
import pandas as pd

hyper_params = {
    'E' : [18,36,54,72,90],
    'F' : [4,8,12,16,20],
    'mode': ['CA', 'simple'],
    'cft_mode': ['channel', 'patch', 'all']
}
output = {}
output['E'] = []
output['F'] = []
output['mode'] = []
output['cft_mode'] = []
output['cost'] = []

# combinations = list(itertools.product(*hyper_params.values()))
# for c in combinations:
#     E = c[0]
#     E_mlp = E * 4
#     F = c[1]
#     mode = c[2]
#     cft_mode = c[3]
#     enc = encoder(P, E, E_mlp, I, F)
#     dec = decoder(P, E, E_mlp, I, F, mode)
#     cft_val = cft(P, E, E_mlp, I, F, cft_mode)
    
#     total = enc + dec + cft_val
#     output['E'].append(E)
#     output['F'].append(F)
#     output['mode'].append(mode)
#     output['cft_mode'].append(cft_mode)
#     output['cost'].append(total)

# df1 = pd.DataFrame(data=output)
    
# '''

import pandas as pd
def calc_cost_optuna():
    df = pd.read_pickle(r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\OPTUNA Expt Results 3 HPC.pkl")
    df1 = pd.read_pickle(r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\OPTUNA Expt Results 3 DOC.pkl")
    for i in range(len(df)):
        E = df.loc[i,'params_Hidden size, E']
        E_mlp = E * 4
        F = df.loc[i,'params_Model feature size, F']
        mode = df.loc[i, 'params_Decode mode']
        cft_mode = df.loc[i, 'params_Cft mode']
 
        
        total = model(C, Cout, I, F, P, E, mode, cft_mode)
        
        df.loc[i, 'cost'] = total
    
    return df

# df1 = calc_cost_optuna()