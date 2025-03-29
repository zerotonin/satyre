#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:12:44 2019

@author: iaji
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

"""      _  __                          _                          
   __| |/ _| __   _____    ___  _ __| |                         
  / _` | |_  \ \ / / __|  / _ \| '__| |                         
 | (_| |  _|  \ V /\__ \ | (_) | |  | |                         
  \__,_|_|     \_/ |___/  \___/|_|  |_|    __                 _ 
  _ __ __ _ _ __   __| | ___  _ __ ___    / _| ___   ___   __| |
 | '__/ _` | '_ \ / _` |/ _ \| '_ ` _ \  | |_ / _ \ / _ \ / _` |
 | | | (_| | | | | (_| | (_) | | | | | | |  _| (_) | (_) | (_| |
 |_|  \__,_|_| |_|\__,_|\___/|_| |_| |_| |_|  \___/ \___/ \__,_|
   __| | __ _ _ __| | __   ___ ___  _ __   __| |                
  / _` |/ _` | '__| |/ /  / __/ _ \| '_ \ / _` |                
 | (_| | (_| | |  |   <  | (_| (_) | | | | (_| |                
  \__,_|\__,_|_|  |_|\_\  \___\___/|_| |_|\__,_|   
""" 




"""
 __  ___  ___  __      __    __  ___ 
/__`  |  |__  |__)    /__` |  / |__  
.__/  |  |___ |       .__/ | /_ |___ 
"""
allStepSize_orl_df = []
allStepSize_orl_df.append(np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/df_randomFood_darkCond1000trials/allStepSize_df.npy'))
allStepSize_orl_df.append(np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allStepSize_df.npy'))

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
ProbDens = []
titles= ['ORL', 'dark fly']
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))
    plt.figure()
    prob, edges,_ = plt.hist(allStepSize_orl_df[i].flatten(), bins = nr_bins, density = True) 
    ProbDens.append(max(prob)) 
    
titles= ['ORL', 'dark fly']
for i in range(len(allStepSize_orl_df)):
    nr_bins = int(round(np.sqrt(len(allStepSize_orl_df[i].flatten()))))
    plt.figure()
    prob, edges,_ = plt.hist(allStepSize_orl_df[i].flatten(), bins = nr_bins, density = True) 
    ProbDens.append(max(prob))
    maxY = max(ProbDens)+(0.1*max(ProbDens))
    midPoints = []
    for j in range(len(edges)-1):
        midPoints.append((edges[j] + edges[j+1]) / 2)
    plt.errorbar(midPoints,prob,yerr=[upperCI_allParams[i],lowerCI_allParams[i]], alpha=0.5, ecolor = 'green')
    ax = plt.gca()
    ax.tick_params(axis = 'both', labelsize = 15)
    ax.set_ylim(0,maxY)
    ax.set_ylabel('Probability density', fontsize = 18)    
    ax.set_xlabel('Step size [mm]', fontsize = 18)
    ax.set_title(titles[i], fontsize = 22) 
    
"""
      ___       __   __    ___    ___  __                __                __        ___  __  
\  / |__  |    /  \ /  ` |  |  | |__  /__`     /\  |\ | |  \     /\  |\ | / _` |    |__  /__` 
 \/  |___ |___ \__/ \__, |  |  | |___ .__/    /~~\ | \| |__/    /~~\ | \| \__> |___ |___ .__/
"""

allThrustVelocities_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allThrustVelocities_df.npy')
allThrustVelocities_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allThrustVelocities.npy')

allSlipVelocities_df=np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/df_randomFood_darkCond1000trials/allSlipVelocities_df.npy')
allSlipVelocities_orl=np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allSlipVelocities_df.npy')

allYawVelocities_orl_df = []
allYawVelocities_orl_df.append(np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/df_randomFood_darkCond1000trials/allYawVelocities_df.npy'))
allYawVelocities_orl_df.append(np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allYawVelocities_df.npy'))

allYawAngles_orl_df = []
allYawAngles_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allYawAngles_df.npy')
allYawAngles_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allYawAngles.npy')

allPsiAngles_orl_df = []
allPsiAngles_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allPsiAngles_df.npy')
allPsiAngles_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allPsiAngles.npy')



def findMargin_of_Error(allThrustVelocities_df):
    zscore = 1.96
    nrTrials = 1000
    nr_bins = int(round(np.sqrt(len(allThrustVelocities_df.flatten()))))  
    hist_allTrials = np.ones((nrTrials, nr_bins))*np.nan
    for k in range(nrTrials):
        hist,edges = np.histogram(allThrustVelocities_df[k], nr_bins, density=True)
        hist_allTrials[k,:] = hist
    mean_histAllTrials = np.mean(hist_allTrials, axis=0)
    margin_of_error = zscore*stats.sem(hist_allTrials, axis=0)
    return margin_of_error, mean_histAllTrials, edges

def findMaxY(allThrustVelocities_df):
    prob,edges = np.histogram(allThrustVelocities_df.flatten(), bins='fd', density = True)
    return max(prob)

    
def plotVelocityDistribution(allThrustVelocities_df, MaxProbDens, i, title='dark fly'):
    dictKeys = ['thrust velocities', 'slip velocities', 'yaw velocities', 'yaw angles', 'psi angles']
    units = ['mm/s', 'mm/s', 'rad/s', 'rad', 'rad']
#    nr_bins = int(round(np.sqrt(len(allThrustVelocities_df.flatten()))))
    plt.hist(allThrustVelocities_df.flatten(), bins = 'fd', density = True)
    maxY = max(MaxProbDens)
#    midPoints = 0.5*(edges[1:]+edges[:-1])
#    plt.bar(midPoints, prob, color='r', yerr=margin_of_error, capsize=2)
#    plt.errorbar(midPoints,prob,yerr=margin_of_error, alpha=0.5, ecolor='green')
    ax = plt.gca()
    ax.set_ylim(0, maxY+0.05*maxY)
    ax.set_ylabel('Probability density', fontsize = 36)   
    ax.set_xlabel('%s (%s)' %(dictKeys[i], units[i]), fontsize = 36)
    ax.set_title(title, fontsize = 60)
    ax.tick_params(axis = 'both', labelsize = 22.5)

MaxProbDens = [findMaxY(allYawAngles_df), findMaxY(allYawAngles_orl)]
plt.figure()
plotVelocityDistribution(allYawAngles_df, MaxProbDens, 3, title='dark-fly') 

plt.figure()
plt.hist(allThrustVelocities_df.flatten(), bins = 'fd', density = True)

title = title
dictKeys = ['thrust velocities', 'slip velocities', 'yaw velocities', 'yaw angles', 'psi angles']
units = ['mm/s', 'mm/s', 'deg/s', 'deg', 'deg']
midPoints = 0.5*(edges[1:]+edges[:-1])
plt.bar(midPoints, meanHist, color='r', yerr=margin_of_error, capsize=2)
#    plt.errorbar(midPoints,prob,yerr=margin_of_error, alpha=0.5, ecolor='green')
ax = plt.gca()
ax.set_ylim(0, maxY+0.1*maxY)
ax.set_ylabel('Probability density', fontsize = 18)   
ax.set_xlabel('%s (%s)' %(dictKeys[i], units[i]), fontsize = 18)
ax.set_title(title, fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 15)


fig, ax = plt.subplots(1,2)
ax[0].set_title('wt')
ax[0].
plt.figure()
plt.boxplot(allThrustVelocities_orl.flatten())

ax[1].set_title('dark-fly')
ax[1].
plt.boxplot(allThrustVelocities_df.flatten())


