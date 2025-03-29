#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:29:48 2019

@author: iaji
"""
import numpy as np
import scipy.stats as stats

allPerformedPMs_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allPerformedPMS.npy')
allPerformedPMs_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allPerformedPMs.npy')

VELO_arr_df = np.load('/home/iaji/Documents/codes/rawData/VELO_arr_df.npy')
VELO_arr_orl = np.load('/home/iaji/Documents/codes/rawData/VELO_arr_orl.npy')

def calculateTransVsRot(VELO_arr, all_performedPMs):
    threshold_transVel = 0.5 #mm/s
    threshold_rotVel = np.deg2rad(100) #rad/s        
    types_of_movements = []
    all_performedPMs = all_performedPMs.astype(int)
    for i in range(len(all_performedPMs)):
        thrustVel = abs(VELO_arr[all_performedPMs[i]][0])
        slipVel = abs(VELO_arr[all_performedPMs[i]][1])
        rotVel = abs(VELO_arr[all_performedPMs[i]][2])
        if thrustVel + slipVel <= threshold_transVel and rotVel <= threshold_rotVel:
            types_of_movements.append(0)
        elif thrustVel + slipVel > threshold_transVel and rotVel <= threshold_rotVel:
            types_of_movements.append(1)
        else:
            types_of_movements.append(2)        
    return np.array(types_of_movements)

PMTypes_performed_df = np.ones((1000, 150002))*np.nan
for i in range(len(allPerformedPMs_df)):  
    PMTypes_performed_df[i,:] = calculateTransVsRot(VELO_arr_df, allPerformedPMs_df[i])

movementProportion_df = np.ones((1000,3))*np.nan
for i in range(len(PMTypes_performed_df)):
    nrPMsPerformed = PMTypes_performed_df.shape[1]
    unique, counts = np.unique(PMTypes_performed_df[i], return_counts = True)
    movementProportion_df[i,0] = (counts[0] / nrPMsPerformed )*100
    movementProportion_df[i,1] = (counts[1]/nrPMsPerformed )*100 
    movementProportion_df[i,2] = (counts[2]/nrPMsPerformed )*100 
    
movementProportion_df = np.load('/home/iaji/Documents/codes/movementProportion_df_100deg.npy')
movementProportion_orl = np.load('/home/iaji/Documents/codes/movementProportion_orl_100deg.npy')
np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/ratioR2T_contrastValue_rotationThreshold100deg/contrastValue_df.txt', contrast_df, newline='\n')

ratioR2T_df = movementProportion_df[:,2]/movementProportion_df[:,1] 
ratioR2T_orl = movementProportion_orl[:,2]/movementProportion_orl[:,1] 

contrast_df = (movementProportion_df[:,1]-movementProportion_df[:,2])/(movementProportion_df[:,1]+movementProportion_df[:,2])
contrast_orl = (movementProportion_orl[:,1]-movementProportion_orl[:,2])/(movementProportion_orl[:,1]+movementProportion_orl[:,2])
np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/stationaryProportion/stationaryProp_orl.txt', movementProportion_orl[:,0], newline='\n')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#__, pVal = stats.ttest_ind(ratioR2T_orl, ratioR2T_df)
df = pd.DataFrame(data={'orl': ratioR2T_orl[:], 'dark-fly': ratioR2T_df[:]}, index=range(len(ratioR2T_orl)))
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylim([0,4])
ax.set_ylabel('Ratio rotational/translational', fontsize = 26)
ax.set_title('Ratio of rotational to translational \n protoypical movements', fontsize = 36)
#if pVal < 0.001:
#    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
#    y, h, col = df['dark-fly'].max()+0.03*df['dark-fly'].max(), 0.03, 'k'
#    plt.plot([x1, x1, x2, x2], [df['orl'].max()+0.03*df['orl'].max(), y+h, y+h, y], lw=1.5, c=col)
#    plt.text((x1+x2)*.5, y+h, "p-value < 0.001", ha='center', va='bottom', color=col, fontsize = 20)
 
#__, pVal = stats.ttest_ind(contrast_orl, contrast_df)
df = pd.DataFrame(data={'orl': contrast_orl[:], 'dark-fly': contrast_df[:]}, index=range(len(contrast_orl)))
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylim([-0.5,0.2])
ax.set_title('Ratio of rotational to translational \n protoypical movements', fontsize = 36)
ax.set_ylabel('Contrast value', fontsize = 26)
#if pVal < 0.001:
#    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
#    y, h, col = df['dark-fly'].max()+0.03*df['dark-fly'].max(), 0.03, 'k'
#    plt.plot([x1, x1, x2, x2], [df['orl'].max()+0.03*df['orl'].max(), y+h, y+h, y], lw=1.5, c=col)
#    plt.text((x1+x2)*.5, y+h, "p-value < 0.001", ha='center', va='bottom', color=col, fontsize = 20)


df = pd.DataFrame(data={'orl': movementProportion_orl[:,0], 'dark-fly': movementProportion_df[:,0]}, index=range(len(movementProportion_df)))
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_title('Proportion of stationary movements', fontsize = 36)
ax.set_ylabel('Proportion of stationary movements [%]', fontsize = 26)

movementProportion_orl = np.load('/home/iaji/Documents/codes/movementProportion_orl.npy')
movementProportion_df = np.load('/home/iaji/Documents/codes/movementProportion_df.npy')
allMovementTypes_orl_df = [movementProportion_orl, movementProportion_df]
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allMovementTypes_orl_df)):
    margin_of_error = zscore*stats.sem(allMovementTypes_orl_df[i], axis=0)
    upperCI_allParams.append(margin_of_error)
    lowerCI_allParams.append(margin_of_error)
velType = np.arange(3)
maxY = max([max(np.mean(allMovementTypes_orl_df[0], axis=0))+max(upperCI_allParams[0]),
          max(np.mean(allMovementTypes_orl_df[1], axis=0))+max(upperCI_allParams[1])])
titles= ['wt', 'dark-fly']
fig, axs = plt.subplots(1,2)
axs = axs.ravel()
for idx, ax in enumerate(axs):
    ax.bar(velType, 
           (np.mean(allMovementTypes_orl_df[idx], axis=0)),
           yerr=[upperCI_allParams[idx], lowerCI_allParams[idx]], capsize=7, alpha=0.7)
    ax.set_xticks(velType)
    ax.set_xticklabels(('stationary', 'translational', 'rotational'), fontsize = 18)
    ax.set_ylim(0,100)
    ax.set_ylabel('Proportion of movement types (%)', fontsize = 18)
    ax.set_title(titles[idx], fontsize = 18)
    
    
plt.imshow(targetPM_probabilityMatrix)
plt.axis('equal')
plt.colorbar()
plt.xlabel('target PM', fontsize=24)
plt.ylabel('source PM', fontsize=24)