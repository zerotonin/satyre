#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:23:11 2019

@author: iaji
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

headPosX = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_clusteredFood_darkCond/allHeadPositionsX.npy')
headPosY = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_clusteredFood_darkCond/allHeadPositionsY.npy')
tailPosX = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_clusteredFood_darkCond/allTailPositionsX.npy')
tailPosY = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_clusteredFood_darkCond/allTailPositionsY.npy')

headPosX = headPosX[:,:1000]
headPosY = headPosY[:,:1000]
tailPosX = tailPosX[:,:1000]
tailPosY = tailPosY[:,:1000]

from shapely.geometry import Polygon
nrTrials = 1000
areaCovered_allTrials = np.ones((nrTrials))*np.nan
areaCovered_oneTrial = np.ones((1000-1))*np.nan
for i in range(nrTrials):
    for j in range(1000-1):
        polygon = Polygon([(headPosX[i,j],headPosY[i,j]), 
                       (tailPosX[i,j], tailPosY[i,j]), 
                       (tailPosX[i,j+1], tailPosY[i,j+1]), 
                       (headPosX[i,j+1], headPosY[i,j+1])])
        areaCovered_oneTrial[j] = polygon.area
        areaCovered_allTrials[i] = np.sum(areaCovered_oneTrial)
        
areaCovered_allTrials_orl = np.load('/home/iaji/Documents/codes/areaCovered_allTrials_orl.npy')
areaCovered_allTrials_df = np.load('/home/iaji/Documents/codes/areaCovered_allTrials_df.npy')

#areaCovered_allTrials_rW = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_randomFood_darkCond/allAreaCovered.npy')
#areaCovered_allTrials_LORL = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_randomFood_darkCond/allAreaCovered.npy')
#areaCovered_allTrials_LDF = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_randomFood_darkCond/allAreaCovered.npy')        
#
#areaCovered_allTrials_rW = np.sum(areaCovered_allTrials_rW[:5000], axis=1)      
#areaCovered_allTrials_LORL = np.sum(areaCovered_allTrials_LORL[:5000], axis=1)
#areaCovered_allTrials_LDF = np.sum(areaCovered_allTrials_LDF[:5000], axis=1) 

allStepSize_rW = np.load('/home/iaji/Documents/codes/allStepSize_rW_new.npy')
allStepSize_LORL = np.load('/home/iaji/Documents/codes/allStepSize_LORL_corrected.npy')
allStepSize_LDF  = np.load('/home/iaji/Documents/codes/allStepSize_LDF_corrected.npy')

bodyWidth = 2
areaCovered_perTraject_rW = allStepSize_rW[:,:1000]*bodyWidth
areaCovered_perTraject_LORL = allStepSize_LORL[:,:1000]*bodyWidth
areaCovered_perTraject_LDF = allStepSize_LDF[:,:1000]*bodyWidth

areaCovered_allTrials_rW = np.sum(areaCovered_perTraject_rW, axis=1)
areaCovered_allTrials_LORL = np.sum(areaCovered_perTraject_LORL, axis=1)
areaCovered_allTrials_LDF = np.sum(areaCovered_perTraject_LDF, axis=1)

#Without Tohoku drift
areaCovered_perTraject_orl = allStepSize_orl[:,:1000]*bodyWidth
areaCovered_perTraject_df = allStepSize_df[:,:1000]*bodyWidth

areaCovered_allTrials_orl_noTohoku = np.sum(areaCovered_perTraject_orl, axis=1)
areaCovered_allTrials_df_noTohoku = np.sum(areaCovered_perTraject_df, axis=1)

#np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/areaCovered_10s/areaCovered_rW.txt', areaCovered_allTrials_rW, newline='\n')

import pandas as pd
import seaborn as sns
d_all={'random walk': areaCovered_allTrials_rW,
   'Levy ORL':areaCovered_allTrials_LORL,
   'Levy DF': areaCovered_allTrials_LDF,                    
   'ORL': areaCovered_allTrials_orl, 
   'dark-fly': areaCovered_allTrials_df}
#d={'orl': areaCovered_allTrials_orl_1KSteps, 
#   'dark-fly': areaCovered_allTrials_df_1KSteps}
df = pd.DataFrame.from_dict(d_all, orient='index')
df=df.T
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
#ax.set(yscale='log')
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylabel('Area covered per simulation [$mm^2$]', fontsize = 24) 
ax.set_title('Total area covered by mechanosensory field \n in 10 s', fontsize = 36)
#if pVal < 0.001:
#    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
#    y, h, col = df['dark-fly'].max() + 2, 2, 'k'
#    plt.plot([x1, x1, x2, x2], [df['wt'].max()+2, y+h, y+h, y], lw=1.5, c=col)
#    plt.text((x1+x2)*.5, y+h, "p-value < 0.001", ha='center', va='bottom', color=col, fontsize = 20)




#plot tohoku vs no tohoku
#np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/areaCovered_noDrifts/areaCovered_df_noDrifts.txt', areaCovered_allTrials_df_noTohoku, newline='\n')
d_all={'ORL': areaCovered_allTrials_orl,
   'ORL without drifts':areaCovered_allTrials_orl_noTohoku,                      
   'dark-fly': areaCovered_allTrials_df,
   'dark-fly without drifts': areaCovered_allTrials_df_noTohoku}
#d={'orl': areaCovered_allTrials_orl_1KSteps, 
#   'dark-fly': areaCovered_allTrials_df_1KSteps}
df = pd.DataFrame.from_dict(d_all, orient='index')
df=df.T
plt.figure()
sns.boxplot(data=df, notch=True)
ax = plt.gca()
#ax.set(yscale='log')
ax.tick_params(axis = 'both', labelsize = 24)
ax.set_ylabel('Area covered per simulation [$mm^2$]', fontsize = 24) 
ax.set_title('Total area covered by mechanosensory field \n in 10 s', fontsize = 36)