#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:16:17 2019

@author: iaji
"""

import numpy as np
import matplotlib.pyplot as plt

allStepSize_rW = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_clusteredFood_darkCond/allStepSize.npy')
allStepSize_LORL = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_clusteredFood_darkCond/allStepSize.npy')
allStepSize_LDF = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_clusteredFood_darkCond/allStepSize.npy')
allStepSize_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allStepSize.npy')
allStepSize_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_darkCond/allStepSize_df.npy')

allStepSize_orl100k = np.random.choice(allStepSize_orl.flatten(), size=100000, replace=False)
allStepSize_df100k = np.random.choice(allStepSize_df.flatten(), size=100000, replace=False)
#np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/stepSize/stepSize_allTrials_df.txt', allStepSize_df, delimiter=',', newline='\n')

#Only for notConstrained simulations of levy df and levy orl
all_LORL = allStepSize_LORL.flatten()
for idx, ele in enumerate(all_LORL):
    if ele > 0.2:
        all_LORL[idx] = np.nan
stepSize_median = np.nanmedian(all_LORL)
for i,__ in enumerate(allStepSize_LORL):
    for j in range(allStepSize_LORL.shape[1]):
        if allStepSize_LORL[i,j] > 0.2:
            allStepSize_LORL[i,j] = stepSize_median

#Only for notConstrained simulations of random walk
factor = 0.0125/0.1387
allStepSize_rW = factor*allStepSize_rW

allStepSize = [allStepSize_orl100k, allStepSize_df100k]

def findMaxY(inputArray):
    prob,edges = np.histogram(inputArray, bins='fd', density = True)
    return max(prob)

def plotVelocityDistribution(inputArray, maxProbDens, title='dark fly'):
    dictKeys = ['step size']
    units = ['mm']
#    nr_bins = int(round(np.sqrt(len(inputArray.flatten()))))
    plt.hist(inputArray, bins=300, density = True)
    maxY = max(maxProbDens)
#    midPoints = 0.5*(edges[1:]+edges[:-1])
#    plt.bar(midPoints, prob, color='r', yerr=margin_of_error, capsize=2)
#    plt.errorbar(midPoints,prob,yerr=margin_of_error, alpha=0.5, ecolor='green')
    ax = plt.gca()
#    plt.yscale('log')
    ax.set_ylim(0, maxY+0.1*maxY)
    ax.set_ylabel('Probability density', fontsize = 36)   
    ax.set_xlabel('%s (%s)' %(dictKeys[0], units[0]), fontsize = 36)
    ax.set_title(title, fontsize = 36)
    ax.tick_params(axis = 'both', labelsize = 22.5)
    
maxProbDens = [findMaxY(ele) for __,ele in enumerate(allStepSize)]
plt.figure()
plotVelocityDistribution(allStepSize_df100k, maxProbDens, title='Step Size Distribution \n (dark-fly)')


import pandas as pd
import seaborn as sns
__, pVal = stats.ttest_ind(allStepSize_orl, allStepSize_df)
d_all={'random walk': allStepSize_rW.flatten(),
   'Levy ORL':allStepSize_LORL.flatten(),
   'Levy DF': allStepSize_LDF.flatten(),                    
   'ORL': allStepSize_orl.flatten(), 
   'dark-fly': allStepSize_df.flatten()}
d={'ORL': allStepSize_orl100k, 'dark-fly': allStepSize_df100k}
df=pd.DataFrame.from_dict(d, orient='index')
df=df.T
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylabel('Step size [mm]', fontsize = 24) 
ax.set_title('Step size', fontsize = 36)



plt.hist(allStepSize_rW100k, bins=300, density=True)
plt.hist(allStepSize_LORL100k, bins=300, density=True)
plt.hist(allStepSize_LDF100k, bins=300,density=True)
plt.hist(allStepSize_orl100k, bins=300,density=True)
plt.hist(allStepSize_df100k, bins=300,density=True)



#total distance travelled
a = [allStepSize_rW, allStepSize_LORL, allStepSize_LDF, allStepSize_orl, allStepSize_df]
total_distance = np.ones((5))*np.nan
for idx, __ in enumerate (a):
    total_distance[idx] = np.median(np.sum(a[idx], axis=1))

d_all={'random walk': np.sum(allStepSize_rW, axis=1),
   'Levy ORL':np.sum(allStepSize_LORL, axis=1),
   'Levy DF': np.sum(allStepSize_LDF, axis=1),                    
   'ORL': np.sum(allStepSize_orl, axis=1), 
   'dark-fly': np.sum(allStepSize_df, axis=1)}
#np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/totalDistanceTravelled/totalDistanceTravelled_df.txt', d_all['dark-fly'], delimiter=',', newline='\n')
df=pd.DataFrame.from_dict(d_all, orient='index')
df=df.T
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylabel('Distance travelled [mm]', fontsize = 24) 
ax.set_title('Total distance travelled per simulation', fontsize = 36)

