#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:11:22 2019

@author: iaji
"""
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
allFoodFound_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/allFoodFound_df.npy')
allFoodFound_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allFoodFound.npy')
allFoodFound_orl_df = [allFoodFound_orl, allFoodFound_df]
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allFoodFound_orl_df)):
    margin_of_error = zscore*stats.sem(allFoodFound_orl_df[i], axis=0)
    upperCI_allParams.append(float(margin_of_error))
    lowerCI_allParams.append(float(margin_of_error))
maxY = max([np.median(allFoodFound_orl_df[0]), np.median(allFoodFound_orl_df[1])])+max(upperCI_allParams)
plt.figure()
plt.bar(np.arange(len(allFoodFound_orl_df)), [np.median(allFoodFound_orl_df[0]), np.median(allFoodFound_orl_df[1])],
        yerr=[lowerCI_allParams, upperCI_allParams], alpha=0.7, capsize=7)
plt.ylim(0, maxY+0.5*maxY)
plt.xticks(np.arange(len(allFoodFound_orl_df)), ('wt', 'dark-fly'))
ax=plt.gca()
ax.set_ylabel('Average no. of foods collected/5 mins of simulation', fontsize = 20)   
ax.set_title('Efficiency in finding clustered foods', fontsize = 40)
ax.tick_params(axis = 'both', labelsize = 36)
if pVal < 0.05:
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = maxY + 2, 0.2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "p-value < 0.05", ha='center', va='bottom', color=col, fontsize = 22)

__, pVal = stats.ttest_ind(allFoodFound_orl, allFoodFound_df)

import pandas as pd
import seaborn as sns
__, pVal = stats.ttest_ind(allFoodFound_orl, allFoodFound_df)
df = pd.DataFrame(data={'wt': allFoodFound_orl[:,0], 'dark-fly': allFoodFound_df[:,0]}, index=range(len(allFoodFound_orl)))
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.set_xticks(np.arange(len(allFoodFound_orl_df)), ('wt', 'dark-fly'))
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylabel('No. of foods collected', fontsize = 24) 
ax.set_title('Efficiency in finding randomly dispersed foods \n (light condition)', fontsize = 36)
if pVal < 0.001:
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = df['dark-fly'].max() + 2, 2, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "p-value < 0.05", ha='center', va='bottom', color=col, fontsize = 22)


allPerformedPMs_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allPerformedPMS.npy')
allPerformedPMs_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allPerformedPMs.npy')

def calculateTransVsRot(VELO_arr, all_performedPMs):
    threshold_transVel = 0.5 #mm/s
    threshold_rotVel = 1.6 #rad/s        
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

ratioR2T_df = movementProportion_df[:,2]/movementProportion_df[:,1] 
ratioR2T_orl = movementProportion_orl[:,2]/movementProportion_orl[:,1] 
import pandas as pd
import seaborn as sns
__, pVal = stats.ttest_ind(ratioR2T_orl, ratioR2T_df)
df = pd.DataFrame(data={'wt': ratioR2T_orl[:], 'dark-fly': ratioR2T_df[:]}, index=range(len(ratioR2T_orl)))
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_title('Ratio of rotational to translational \n protoypical movements', fontsize = 36)
if pVal < 0.001:
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = df['dark-fly'].max()+0.03*df['dark-fly'].max(), 0.03, 'k'
    plt.plot([x1, x1, x2, x2], [df['wt'].max()+0.03*df['wt'].max(), y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "p-value < 0.001", ha='center', va='bottom', color=col, fontsize = 20)

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
    
#def calcBoutDuration(movementType, PMTypes_performed):
#    PMTypes_performed = np.append(PMTypes_performed, np.nan)
#    BoutCounts = []
#    count = 0
#    for i,x in enumerate(PMTypes_performed):
#        if x == movementType:
#            count += 1
#        else: 
#            if count > 0:
#                BoutCounts.append(count)
#            count = 0
#    return BoutCounts
#
#MedianBoutDuration_df = np.ones((3))*np.nan
#BoutCounts_stat = calcBoutDuration(0, PMTypes_performed_df.flatten())
#BoutCounts_trans_orl = calcBoutDuration(1, PMTypes_performed_orl.flatten())
#BoutCounts_rot = calcBoutDuration(2, PMTypes_performed_df.flatten())
#MedianBoutDuration_df[0] = np.mean(BoutCounts_stat)
#MedianBoutDuration_df[1] = np.mean(BoutCounts_trans)
#MedianBoutDuration_df[2] = np.mean(BoutCounts_rot)

plt.boxplot([BoutCounts_trans_orl, BoutCounts_trans_df])

stats.f_oneway(movementProportion_df[:,0], movementProportion_df[:,1], movementProportion_df[:,2])

