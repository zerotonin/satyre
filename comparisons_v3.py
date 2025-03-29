#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:33:40 2019

@author: iaji
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

file_dir = os.path.dirname('walkSimOO_updated_darkFly')
sys.path.append('/home/iaji/Documents/codes')

"""
  ____                    _                 _       _   _                
 |  _ \ _   _ _ __    ___(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __  _  
 | |_) | | | | '_ \  / __| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \(_) 
 |  _ <| |_| | | | | \__ \ | | | | | | |_| | | (_| | |_| | (_) | | | |_  
 |_| \_\\__,_|_| |_| |___/_|_|_|_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_(_) 
 
 | |    _____   ___   _   / _| (_) __ _| |__ | |_                        
 | |   / _ \ \ / / | | | | |_| | |/ _` | '_ \| __|                       
 | |__|  __/\ V /| |_| | |  _| | | (_| | | | | |_                        
 |_____\___| \_/  \__, | |_| |_|_|\__, |_| |_|\__|                       
                  |___/           |___/
"""
from walkSimOO_updated_LevyRandom_expBased_v2 import walkSim
modes = ['uniform', 'levy orl', 'levy df']
maxTrial = 50
allFoodFound_orl_df= []
allAreaCovered_orl_df = []
allStepSize_orl_df = []
for i in range(len(modes)):
    ws=walkSim(cauchyAlpha =1.5, nrTrial = maxTrial,
                border = 10000.0, mode=modes[i],maxSteps = 1000,totalFood = 1000,foodMode='clustered')       
    ws.simulateMultiple()
    allFoodFound_orl_df.append(ws.foodFound_total_allTrials)
    allAreaCovered_orl_df.append(ws.areaCovered_allTrials)
    allStepSize_orl_df.append(ws.stepSize_allTrials)

"""
  ____  _       _                     _                                              
 |  _ \| | ___ | |_    __ _ _ __   __| |   ___ ___  _ __ ___  _ __   __ _ _ __ ___ _ 
 | |_) | |/ _ \| __|  / _` | '_ \ / _` |  / __/ _ \| '_ ` _ \| '_ \ / _` | '__/ _ (_)
 |  __/| | (_) | |_  | (_| | | | | (_| | | (_| (_) | | | | | | |_) | (_| | | |  __/_ 
 |_|   |_|\___/ \__|  \__,_|_| |_|\__,_|  \___\___/|_| |_| |_| .__/ \__,_|_|  \___(_)
"""

"""
 __  ___  ___  __      __    __  ___ 
/__`  |  |__  |__)    /__` |  / |__  
.__/  |  |___ |       .__/ | /_ |___ 
"""
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))  
    hist_allTrials = np.ones((nrTrials, nr_bins))*np.nan
    for k in range(nrTrials):
        hist,__ = np.histogram(allStepSize_orl_df[i][k], nr_bins, density=True)
        hist_allTrials[k,:] = hist
    margin_of_error = zscore*stats.sem(hist_allTrials, axis=0)
    upperCI_allParams.append(margin_of_error)
    lowerCI_allParams.append(margin_of_error)  
titles= ['random walk', 'ORL-based Levy', 'DF-based Levy']
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))
    plt.figure()
    prob, edges,_ = plt.hist(allStepSize_orl_df[i].flatten(), bins = nr_bins, density = True) 
    midPoints = []
    for j in range(len(edges)-1):
        midPoints.append((edges[j] + edges[j+1]) / 2)
    plt.errorbar(midPoints,prob,yerr=[upperCI_allParams[i],lowerCI_allParams[i]], alpha=0.5)
    plt.ylabel('Probability density')    
    plt.xlabel('Step size [mm]')
    plt.title(titles[i])  

"""
 ___  __   __   __      ___  __             __  
|__  /  \ /  \ |  \    |__  /  \ |  | |\ | |  \ 
|    \__/ \__/ |__/    |    \__/ \__/ | \| |__/
"""    
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allFoodFound_orl_df)):
    margin_of_error = zscore*stats.sem(allFoodFound_orl_df[i], axis=0)
    upperCI_allParams.append(float(margin_of_error))
    lowerCI_allParams.append(float(margin_of_error))
maxY = max([np.mean(allFoodFound_orl_df[0]), np.mean(allFoodFound_orl_df[1]), np.mean(allFoodFound_orl_df[2])])+max(upperCI_allParams)
plt.figure()
plt.bar(np.arange(len(allFoodFound_orl_df)), 
        [np.mean(allFoodFound_orl_df[0]), np.mean(allFoodFound_orl_df[1]), np.mean(allFoodFound_orl_df[2])],
        yerr=[lowerCI_allParams, upperCI_allParams], alpha=0.7, capsize=7)
plt.ylim(0, maxY+0.1*maxY)
plt.xticks(np.arange(len(allFoodFound_orl_df)), ('random walk', 'ORL-based Levy', 'DF-based Levy'))
ax=plt.gca()
ax.set_ylabel('Average no. of foods collected', fontsize = 18)   
ax.set_title('Efficiency in finding randomly dispersed foods', fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 15)

"""
      __   ___          __   __        ___  __   ___  __  
 /\  |__) |__   /\     /  ` /  \ \  / |__  |__) |__  |  \ 
/~~\ |  \ |___ /~~\    \__, \__/  \/  |___ |  \ |___ |__/
"""
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allAreaCovered_orl_df)):
    margin_of_error = zscore*stats.sem(allAreaCovered_orl_df[i], axis=0)
    upperCI_allParams.append(margin_of_error)
    lowerCI_allParams.append(margin_of_error)   
maxY = max([max(np.mean(allAreaCovered_orl_df[0], axis=0))+max(allAreaCovered_orl_df[0]),
          max(np.mean(allAreaCovered_orl_df[1], axis=0))+max(allAreaCovered_orl_df[1]),
          max(np.mean(allAreaCovered_orl_df[2], axis=0))+max(allAreaCovered_orl_df[2])])
plt.figure()
plt.bar(np.arange(len(allAreaCovered_orl_df)), 
        [np.mean(allAreaCovered_orl_df[0]), np.mean(allAreaCovered_orl_df[1]), np.mean(allAreaCovered_orl_df[1])],
        yerr=[lowerCI_allParams, upperCI_allParams], alpha=0.7, capsize=7)
plt.ylim(0, maxY+0.1*maxY)
plt.xticks(np.arange(len(allAreaCovered_orl_df)), ('random walk', 'ORL-based Levy', 'DF-based Levy'))
ax=plt.gca()
ax.set_ylabel("Average area covered by fly's body", fontsize = 18)   
ax.set_title('Exploration rate of wt-fly vs dark fly', fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 15)

"""
  _                    _       _       _        
 | |    ___   __ _  __| |   __| | __ _| |_ __ _ 
 | |   / _ \ / _` |/ _` |  / _` |/ _` | __/ _` |
 | |__| (_) | (_| | (_| | | (_| | (_| | || (_| |
 |_____\___/ \__,_|\__,_|  \__,_|\__,_|\__\__,_|
"""
targetPM_cumsum_probabilityMatrix_df = np.load('/home/iaji/Documents/codes/rawData/targetPM_cumsum_probabilityMatrix_df.npy')
VELO_arr_df = np.load('/home/iaji/Documents/codes/rawData/VELO_arr_df.npy')
PM_index_df = np.load('/home/iaji/Documents/codes/rawData/PM_index_df.npy')
targetPM_cumsum_probabilityMatrix_orl = np.load('/home/iaji/Documents/codes/rawData/targetPM_cumsum_probabilityMatrix_orl.npy')
VELO_arr_orl = np.load('/home/iaji/Documents/codes/rawData/VELO_arr_orl.npy')
PM_index_orl = np.load('/home/iaji/Documents/codes/rawData/PM_index_orl.npy')

rawDataList = [targetPM_cumsum_probabilityMatrix_orl, PM_index_orl, VELO_arr_orl, 
               targetPM_cumsum_probabilityMatrix_df, PM_index_df, VELO_arr_df]

"""
  ____                    _                 _       _   _               
 |  _ \ _   _ _ __    ___(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __  _ 
 | |_) | | | | '_ \  / __| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \(_)
 |  _ <| |_| | | | | \__ \ | | | | | | |_| | | (_| | |_| | (_) | | | |_ 
 |_|_\_\\__,_|_| |_| |___/_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_(_)
 
 |  _ \|  \/  |     | |__   __ _ ___  ___  __| |                        
 | |_) | |\/| |_____| '_ \ / _` / __|/ _ \/ _` |                        
 |  __/| |  | |_____| |_) | (_| \__ \  __/ (_| |                        
 |_|   |_|  |_|     |_.__/ \__,_|___/\___|\__,_|                        
                                                    
"""
import time
time_start = time.clock()
from walkSimOO_updated_darkFly_multiTrials import walkSim_darkFly
maxTrial = 100
i=3
ws=walkSim_darkFly(rawDataList[i], rawDataList[i+1], rawDataList[i+2], nrTrial = maxTrial,
            border = 10000,maxSteps = 10000,totalFood = 1000,foodMode='random')       
ws.simulateMultiple()   
allFoodFound_df = ws.foodFound_total_allTrials
allPositionCoordsX_df = ws.positionCoordsX_allTrials
allPositionCoordsY_df = ws.positionCoordsY_allTrials
allStepSize_df = ws.stepSize_allTrials
allThrustVelocities_df = ws.thrustVelocities_allTrials
allSlipVelocities_df = ws.slipVelocities_allTrials
allYawVelocities_df = ws.yawVelocities_allTrials
allYawAngles_df = ws.yawAngles_allTrials
allPsiAngles_df = ws.psiAngles_allTrials
allTailPositionsX_df = ws.tailCoordsX_allTrials
allTailPositionsY_df = ws.tailCoordsY_allTrials
allHeadPositionsX_df = ws.headCoordsX_allTrials
allHeadPositionsY_df = ws.headCoordsY_allTrials
time_elapsed = (time.clock() - time_start)

"""
  ____  _       _                     _                                              
 |  _ \| | ___ | |_    __ _ _ __   __| |   ___ ___  _ __ ___  _ __   __ _ _ __ ___ _ 
 | |_) | |/ _ \| __|  / _` | '_ \ / _` |  / __/ _ \| '_ ` _ \| '_ \ / _` | '__/ _ (_)
 |  __/| | (_) | |_  | (_| | | | | (_| | | (_| (_) | | | | | | |_) | (_| | | |  __/_ 
 |_|   |_|\___/ \__|  \__,_|_| |_|\__,_|  \___\___/|_| |_| |_| .__/ \__,_|_|  \___(_)
"""

"""
 __  ___  ___  __      __    __  ___ 
/__`  |  |__  |__)    /__` |  / |__  
.__/  |  |___ |       .__/ | /_ |___ 
"""
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))  
    hist_allTrials = np.ones((nrTrials, nr_bins))*np.nan
    for k in range(nrTrials):
        hist,__ = np.histogram(allStepSize_orl_df[i][k], nr_bins, density=True)
        hist_allTrials[k,:] = hist
    margin_of_error = zscore*stats.sem(hist_allTrials, axis=0)
    upperCI_allParams.append(margin_of_error)
    lowerCI_allParams.append(margin_of_error)  
titles= ['ORL', 'DF']
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))
    plt.figure()
    prob, edges,_ = plt.hist(allStepSize_orl_df[i].flatten(), bins = nr_bins, density = True) 
    midPoints = []
    for j in range(len(edges)-1):
        midPoints.append((edges[j] + edges[j+1]) / 2)
    plt.errorbar(midPoints,prob,yerr=[upperCI_allParams[i],lowerCI_allParams[i]], alpha=0.5)
    plt.ylabel('Probability density')    
    plt.xlabel('Step size [mm]')
    plt.title(titles[i])  

"""
 ___  __   __   __      ___  __             __  
|__  /  \ /  \ |  \    |__  /  \ |  | |\ | |  \ 
|    \__/ \__/ |__/    |    \__/ \__/ | \| |__/
"""    
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allFoodFound_orl_df)):
    margin_of_error = zscore*stats.sem(allFoodFound_orl_df[i], axis=0)
    upperCI_allParams.append(float(margin_of_error))
    lowerCI_allParams.append(float(margin_of_error))
maxY = max([np.mean(allFoodFound_orl_df[0]), np.mean(allFoodFound_orl_df[1])])+max(upperCI_allParams)
plt.figure()
plt.bar(np.arange(len(allFoodFound_orl_df)), [np.mean(allFoodFound_orl_df[0]), np.mean(allFoodFound_orl_df[1])],
        yerr=[lowerCI_allParams, upperCI_allParams], alpha=0.7, capsize=7)
plt.ylim(0, maxY+0.1*maxY)
plt.xticks(np.arange(len(allFoodFound_orl_df)), ('wt', 'DF'))
ax=plt.gca()
ax.set_ylabel('Average no. of foods collected', fontsize = 18)   
ax.set_title('Efficiency in finding randomly dispersed foods', fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 15)

"""
      __   ___          __   __        ___  __   ___  __  
 /\  |__) |__   /\     /  ` /  \ \  / |__  |__) |__  |  \ 
/~~\ |  \ |___ /~~\    \__, \__/  \/  |___ |  \ |___ |__/
"""
zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(allAreaCovered_orl_df)):
    margin_of_error = zscore*stats.sem(allAreaCovered_orl_df[i], axis=0)
    upperCI_allParams.append(margin_of_error)
    lowerCI_allParams.append(margin_of_error)   
maxY = max([max(np.mean(allAreaCovered_orl_df[0], axis=0))+max(allAreaCovered_orl_df[0]),
          max(np.mean(allAreaCovered_orl_df[1], axis=0))+max(allAreaCovered_orl_df[1])])
plt.figure()
plt.bar(np.arange(len(allAreaCovered_orl_df)), [np.mean(allAreaCovered_orl_df[0]), np.mean(allAreaCovered_orl_df[1])],
        yerr=[lowerCI_allParams, upperCI_allParams], alpha=0.7, capsize=7)
plt.ylim(0, maxY+0.1*maxY)
plt.xticks(np.arange(len(allAreaCovered_orl_df)), ('wt', 'DF'))
ax=plt.gca()
ax.set_ylabel("Average area covered by fly's body", fontsize = 18)   
ax.set_title('Exploration rate of wt-fly vs dark fly', fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 15)

"""
       __        ___        ___      ___    ___      __   ___  __  
 |\/| /  \ \  / |__   |\/| |__  |\ |  |      |  \ / |__) |__  /__` 
 |  | \__/  \/  |___  |  | |___ | \|  |      |   |  |    |___ .__/
"""
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
titles= ['ORL', 'DF']
fig, axs = plt.subplots(1,2)
axs = axs.ravel()
for idx, ax in enumerate(axs):
    ax.bar(velType, 
           (np.mean(allMovementTypes_orl_df[idx], axis=0)),
           yerr=[upperCI_allParams[idx], lowerCI_allParams[idx]], capsize=7, alpha=0.7)
    ax.set_xticks(velType)
    ax.set_xticklabels(('stationary', 'translational', 'rotational'))
    ax.set_ylim(0,100)
    ax.set_ylabel('Proportion of movement types (%)')
    ax.set_title(titles[idx])

"""
      ___       __   __    ___    ___  __                __                __        ___  __  
\  / |__  |    /  \ /  ` |  |  | |__  /__`     /\  |\ | |  \     /\  |\ | / _` |    |__  /__` 
 \/  |___ |___ \__/ \__, |  |  | |___ .__/    /~~\ | \| |__/    /~~\ | \| \__> |___ |___ .__/
"""
velocities_angles_dict = {'thrust velocities': allThrustVelocities_orl_df,
                     'slip velocities': allSlipVelocities_orl_df,
                     'yaw velocities': allYawVelocities_orl_df,
                     'yaw angles': allYawAngles_orl_df,
                     'psi angles': allPsiAngles_orl_df}
dictKeys = ['thrust velocities', 'slip velocities', 'yaw velocities', 'yaw angles', 'psi angles']
units = ['mm/s', 'mm/s', 'rad/s', 'rad', 'rad']
import pandas as pd
velocities_angles_df = pd.DataFrame.from_dict(velocities_angles_dict, orient='index', columns=['orl', 'df'])
velocities_angles_list = [allThrustVelocities_orl_df,
                     allSlipVelocities_orl_df,
                     allYawVelocities_orl_df,
                     allYawAngles_orl_df,
                     allPsiAngles_orl_df]

zscore = 1.96
upperCI_allParams = []
lowerCI_allParams = []
nrTrials = 50
for i in range(len(velocities_angles_list)):
    for j in range(2):
        nr_bins = int(round(np.sqrt(len(velocities_angles_list[i][j].flatten()))))  
        hist_allTrials = np.ones((nrTrials, nr_bins))*np.nan
        for k in range(nrTrials):
            hist,__ = np.histogram(velocities_angles_list[i][j][k], nr_bins, density=True)
            hist_allTrials[k,:] = hist
        margin_of_error = zscore*stats.sem(hist_allTrials, axis=0)
        upperCI_allParams.append(np.mean(hist_allTrials, axis=0) + margin_of_error)
        lowerCI_allParams.append(np.mean(hist_allTrials, axis=0) - margin_of_error)

def plotVelocityDistribution(i, velocities_angles_list):
    i=i
    fig, axs = plt.subplots(1,2)
    axs = axs.ravel()
    title = ['wt', 'dark fly']
    ProbDens = []
    dictKeys = ['thrust velocities', 'slip velocities', 'yaw velocities', 'yaw angles', 'psi angles']
    units = ['mm/s', 'mm/s', 'deg/s', 'deg', 'deg']
    for idx, ax in enumerate(axs):
        nr_bins = int(round(np.sqrt(len(velocities_angles_list[i][idx].flatten()))))
        prob,edges,_ = axs[idx].hist(velocities_angles_list[i][idx].flatten(), nr_bins, density = True)
        ProbDens.append(max(prob))
        maxY = max(ProbDens)+(1*max(ProbDens))
        midPoints = []
        for j in range(len(edges)-1):
            midPoints.append((edges[j] + edges[j+1]) / 2)
        axs[idx].errorbar(midPoints,prob,yerr=[upperCI_allParams[i*2+idx],lowerCI_allParams[i*2+idx]], alpha=0.5)
        axs[idx].set_ylim(0, maxY+0.1*maxY)
        axs[idx].set_ylabel('Probability density', fontsize = 18)   
        axs[idx].set_xlabel('%s (%s)' %(dictKeys[i], units[i]), fontsize = 18)
        axs[idx].set_title(title[idx], fontsize = 18)
        axs[idx].tick_params(axis = 'both', labelsize = 15)

for i in range(5):
    plotVelocityDistribution(i, velocities_angles_list) 



import time
time_start = time.clock()
i=0
from walkSimOO_updated_darkFly_v11 import walkSim_darkFly
ws=walkSim_darkFly(rawDataList[i], rawDataList[i+1], rawDataList[i+2], nrTrial = 3,
            border = 10000,maxSteps = 150000,totalFood = 1000,foodMode='random')       
#ws.simulate()
#time_elapsed2 = (time.clock() - time_start) 


import cProfile
cProfile.run('ws.simulateMultiple()')

