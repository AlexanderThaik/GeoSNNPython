''' ====================================================================
% Author: Khalid Youssef, PhD (2023)
% Email: khyous@iu.edu
% ====================================================================
% Supplemental code for demonstrating how to implement the SNN optimization 
% pipeline for landslide susceptibility modelling
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% PLEASE ACKNOWLEDGE THE EFFORT THAT WENT INTO DEVELOPMENT 
% BY REFERENCING THE PAPER:
%
% K. Youssef, K. Shao, S. Moon & L.-S. Bouchard Landslide susceptibility 
% modeling by interpretable neural network. 
% Communications Earth & Environment 
% https://doi.org/10.1038/s43247-023-00806-5
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% ====================================================================
% ====================================================================
% ====================================================================
% Dependencies:
%
% This code was implemented using MATLA 2021b
% It requires statistics & machine learning toolbox
% 
% *Earlier versions of MATLAB might require the deep learning toolbox
% ====================================================================
% Hardware requirements:
%
% This has been successfully tested with a minimum of 64GB RAM using 
% 4 workers, and took approximately 90 minutes to run under these settings
% with an i7-11800H Intel CPU.
% If more RAM is available, the number of workers can be increased for
% faster processing. 
% If you have less than 64GB RAM, you can try reducing the number of
% workers.
% To set the number of workers, use the command parpool(n) before you run 
% this script, where n is the number of workers.
% ====================================================================
% Optimization instructions:
%
% Download the dataset "GSNN_Demo_Data.mat" from:
% https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/D5QPUA.
%
% Run this file for the full pipeline, or call the funcions described
% herein individually.
%
% % Data preparation GSNN_Data_Prep. Loads the dataset and prepares the
% data for optimization. Select the included dataset "GSNN_Demo_Data.mat" 
% for this demonstration. See documentation for how to prepare a dataset 
% for your own data. 
%
% % Tournament ranking is performed in two steps: 
% The function GSNN_TR1 for backwards elimination, and the function 
% GSNN_TR2 for forward selection
%
% % Teacher MST network training is performed by calling the function:
% GSNN_MST 
%
% % SNN network training is performed by caslling the function:
% GSNN_SNN
%
% % Save the SNN model when the optimization is complete.
%
% --------------------------------------------------------------------
% The optimization is to be performed multiple times with different 
% initial conditions, where the model with the highest AUC is selected.
% --------------------------------------------------------------------
% ====================================================================
% Inference instructions:
%
% The second part of this file demonstrates how to use the SNN model for 
% inference, and how to extract and plot the additive feature functions.
%
% The first popup window is for selecting the saved SNN model
% 
% The second popup window is for select the dataset 
%
% GSNN_Inference_Data_Prep is used for preparing the data for inference.
% This data preparation function does not normalize the data, and keeps the
% features in their original range. 
%
% ====================================================================
% For questions and comments, please email: land.slide.snn@gmail.com
% ====================================================================
%% ===================================================================='''
from tkinter import filedialog
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Main parameter values
composite_level = 2 # composite level options: 1, 2
SNN_iterations = 50 # number of SNN training iterations

# Functions paramater values
NN1 = 5 # number of nurons per layer for Tournament ranking - backwards elimination
NE1 = 50 # number of training epochs for Tournament ranking - backwards elimination
reps1 = 4000 # number of models for Tournament ranking - backwards elimination
thr1 = 0.005 # elimination threshold for Tournament ranking - backwards elimination
NN2 = 8 # number of nurons per layer for Tournament ranking - forward selection
NE2 = 50 # number of training epochs for Tournament ranking - forward selection
reps2 = 3 # number of models per step for Tournament ranking - forward selection

# Data preparation function
print('Data preparation ...')
TR,TAR,VL,TARV,TST,TART,names,nameID,MN,MX = GSNN_Data_Prep(composite_level)
print('Data preparation complete')

# Tournament ranking - backwards elimination
print('Tournament ranking step 1/2 ...')
TR1indx = GSNN_TR1(TR,TAR,VL,TARV,NN1,NE1,reps1,thr1)

# select winning Features
names = names[TR1indx] # feature names
nameID = nameID[TR1indx] # feature name IDs
TR = TR[TR1indx] # Training partition
VL = VL[TR1indx] # Validation partition
TST = TST[TR1indx] # Testing partition
MX = MX[TR1indx] # feature minimums
MN = MN[TR1indx] # feature maximums
print('Tournament ranking step 1/2 complete')

# Tournament ranking - forward selection
print('Tournament ranking step 2/2 ...')
TR2indx = GSNN_TR2(TR,TAR,VL,TARV,NN2,NE2,reps2,composite_level)

# select winning Features
names = names[TR2indx]
nameID = nameID[TR2indx]
TR = TR[TR2indx]
VL = VL[TR2indx]
TST = TST[TR2indx]
MX = MX[TR2indx]
MN = MN[TR2indx]
print('Tournament ranking step 2/2 complete')

# Teacher MST network training
print('Teacher MST training ...')
res,resV,resT = GSNN_MST(TR,TAR,VL,TARV,TST,TART)
print('Teacher MST training complete')

# SNN network training
print('SNN training ...')
SNN,ranks = GSNN_SNN(TR,TAR,VL,TARV,TST,TART,res,resV,MN,MX,SNN_iterations)
print('SNN training complete')

file_path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("MATLAB File", "*.mat")], title="Save SNN model")
namesR = names[ranks]
data = {'SNN': SNN, 'namesR': namesR}
sio.savemat(file_path, data)


print('Optimization complete. Click any key to proceed to inference and visualization examples.')
##########################################################################

##########################################################################
######################Inference & Visualization Demo######################
##########################################################################
data = sio.loadmat(file_path)

# Access the loaded variables from the 'data' dictionary
SNN = data['SNN']
namesR = data['namesR']

# Inference data preparation function
composite_level = 2
Features = GSNN_Inference_Data_Prep(composite_level,namesR)

# Inference:
# SNN model parameters
a = SNN.a
b = SNN.b
w = SNN.w
c = SNN.c

# SNN single-sample inference, example:
print('SNN single-sample inference example:')
sample = Features[:, 0]
# individual Features contribution
exponent_term = np.exp(-(a * np.tile(sample, (a.shape[1], 1)) + b) ** 2)
functions = np.sum(w.T * exponent_term, axis=1) + c
print(['------------------'])
print(['------------------'])
for j in range(len(namesR)):
    print(namesR[j] + ': ' + str(sample[0, j]))
    print('f(' + namesR[j] + '): ' + str(functions[j]))
    print('------------------')
# total sample susceptibility
susceptibility = np.sum(np.sum(w.T * np.exp(-(a * np.tile(sample, (1, a.shape[1])) + b) ** 2) + c))
print(['------------------'])
print('susceptibility =', susceptibility)

# SNN batch inference:
susceptibility = np.sum(np.squeeze(np.sum(np.tile(w.T, (1, 1, Features.shape[2])) *
                        np.exp(-((np.tile(a.T, (1, 1, Features.shape[2])) *
                                  np.transpose(np.tile(Features, (1, 1, a.shape[1])), (2, 0, 1)) +
                                  np.tile(b.T, (1, 1, Features.shape[2]))))**2), axis=0)) +
                        np.tile(c, (1, Features.shape[2])))

# SNN batch inference with individual functions:
Functions = np.squeeze(np.sum(np.tile(w.T, (1, 1, Features.shape[2])) *
                    np.exp(-((np.tile(a.T, (1, 1, Features.shape[2])) *
                              np.transpose(np.tile(Features, (1, 1, a.shape[1])), (2, 0, 1)) +
                              np.tile(b.T, (1, 1, Features.shape[2]))))**2), axis=0)) + \
                    np.tile(c, (1, Features.shape[2]))

# Plot feature functions:
d = np.ceil(np.sqrt(len(namesR)))
fig, axes = plt.subplots(d, d, figsize=(12, 12))

for j in range(len(namesR)):
    ax = axes.flatten()[j]
    ax.plot(Features[j], Functions[j], '.')
    tmp = namesR[j]
    ind = np.where(tmp=='&')[0]
    tmp2 = [None, None]
    if len(ind) > 0:
        tmp2[0] = tmp[:ind]
        tmp2[1] = tmp[ind+2:]
    else:
        tmp2 = tmp
    ax.set_title(tmp2)
    mx = np.max(Features[j])
    mn = np.min(Features[j])
    ax.set_xlim([mn, mx])
    ax.set_ylim([0, np.max(Functions)])
    ax.set_xticks([mn, mx])
    ax.set_xticklabels([round(mn, 1), round(mx, 1)], rotation=10)
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.show()
