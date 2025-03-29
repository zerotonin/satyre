#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:25:11 2019

@author: iaji
"""
import numpy as np
import matplotlib.pyplot as plt

allThrustVelocities_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allThrustVelocities_df.npy')
allThrustVelocities_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allThrustVelocities.npy')

allSlipVelocities_df=np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allSlipVelocities_df.npy')
allSlipVelocities_orl=np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allSlipVelocities.npy')

allYawVelocities_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allYawVelocities_df.npy')
allYawVelocities_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allYawVelocities.npy')

allYawAngles_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allYawAngles_df.npy')
allYawAngles_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allYawAngles.npy')

allPsiAngles_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allPsiAngles_df.npy')
allPsiAngles_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allPsiAngles.npy')

#np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/slipVelocities/slipVelocities_df.txt', allSlipVelocities_df, delimiter=',', newline='\n')

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

MaxProbDens = [findMaxY(allYawVelocities_df), findMaxY(allYawVelocities_orl)]
plt.figure()
plotVelocityDistribution(allYawVelocities_orl, MaxProbDens, 2, title='ORL') 


"""BOXPLOT"""
allPsiAngles_orl100k = np.random.choice(allPsiAngles_orl.flatten(), size=100000, replace=False)
allPsiAngles_df100k = np.random.choice(allPsiAngles_df.flatten(), size=100000, replace=False)

def boxplot_velocities(inputArray1, inputArray2, i, title='Thrust velocities'):
    import pandas as pd
    import seaborn as sns
    dictKeys = ['thrust velocities', 'slip velocities', 'yaw velocities', 'yaw angles', 'psi angles']
    units = ['mm/s', 'mm/s', 'rad/s', 'rad', 'rad']
    d={'ORL': inputArray1, 'dark-fly': inputArray2}
    df=pd.DataFrame.from_dict(d, orient='index')
    df=df.T
    plt.figure()
    sns.boxplot(data=df, notch=True)
    ax = plt.gca()
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.set_ylabel('%s (%s)' %(dictKeys[i], units[i]), fontsize = 36)
    ax.set_title(title, fontsize = 36)
boxplot_velocities(allPsiAngles_orl100k, allPsiAngles_df100k, 4, title='Psi angles')
