#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:02:13 2019

@author: iaji
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Carefull these lines only work inside the intranet
"DARK"
allFoodFound_rW = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_randomFood_darkCond/allFoodFound.npy')
allFoodFound_LOrl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_randomFood_darkCond/allFoodFound.npy')
allFoodFound_LDF = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_randomFood_darkCond/allFoodFound.npy')
allFoodFound_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/allFoodFound_df.npy')
allFoodFound_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allFoodFound.npy')

np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/foodFound_randomFood_darkCondition_5minSim/allFoodFound_df.txt', allFoodFound_df, newline='\n')


allFoodFound = [allFoodFound_rW, allFoodFound_LOrl, allFoodFound_LDF, allFoodFound_orl, allFoodFound_df]

import pandas as pd
import seaborn as sns
#__, pVal = stats.f_oneway(allFoodFound_rW, allFoodFound_LOrl, allFoodFound_LDF, allFoodFound_df, allFoodFound_orl)
d_all={'random walk': allFoodFound_rW[:,0],
   'Levy ORL':allFoodFound_LOrl[:,0],
   'Levy DF': allFoodFound_LDF[:,0],                    
   'ORL': allFoodFound_orl[:,0], 
   'dark-fly': allFoodFound_df[:,0]}
d={'ORL': allFoodFound_orl[:,0], 
   'dark-fly': allFoodFound_df[:,0]}
df = pd.DataFrame.from_dict(d_all, orient='index')
df=df.T
plt.figure()

#sns.violinplot(data=df)
#sns.swarmplot(data=df)
sns.boxplot(data=df, notch=True)
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 20)
ax.set_ylabel('No. of foods collected', fontsize = 24) 
ax.set_title('Efficiency in finding randomly dispersed foods \n (dark condition)', fontsize = 36)




"LIGHT"
allFoodFound_rW = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_clusteredFood_lightCond/allFoodFound.npy')
allFoodFound_LOrl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_clusteredFood_lightCond/allFoodFound.npy')
allFoodFound_LDF = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_clusteredFood_lightCond/allFoodFound.npy')
allFoodFound_df = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_clusteredFood_lightCond/allFoodFound.npy')
allFoodFound_orl = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_clusteredFood_lightCond/allFoodFound.npy')

np.savetxt('/media/gwdg-backup/BackUp/Irene/data_for_statisticalTests/foodFound_clusteredFood_lightCondition_5minSim/allFoodFound_orl.txt', allFoodFound_orl, newline='\n')

allFoodFound = [allFoodFound_rW, allFoodFound_LOrl, allFoodFound_LDF, allFoodFound_orl, allFoodFound_df]

import pandas as pd
import seaborn as sns
#__, pVal = stats.f_oneway(allFoodFound_rW, allFoodFound_LOrl, allFoodFound_LDF, allFoodFound_df, allFoodFound_orl)
d={'random walk': allFoodFound_rW[:,0],
   'Levy ORL':allFoodFound_LOrl[:,0],
   'Levy DF': allFoodFound_LDF[:,0],                    
   'ORL': allFoodFound_orl[:,0], 
   'dark-fly': allFoodFound_df[:,0]}
   
df = pd.DataFrame.from_dict(d, orient='index')
df=df.T
#plt.figure()
sns.boxplot(data=df, notch=True,  boxprops=dict(alpha=.3))
#sns.swarmplot(data=df, edgecolor='black')
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 20)
ax.set_ylabel('No. of foods collected', fontsize = 24) 
ax.set_title('Efficiency in finding clustered foods \n (light condition)', fontsize = 36)




"LIGHT AND DARK"
allFoodFound_rW_light = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_randomFood_lightCond/allFoodFound.npy')
allFoodFound_LOrl_light = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_randomFood_lightCond/allFoodFound.npy')
allFoodFound_LDF_light = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_randomFood_lightCond/allFoodFound.npy')
allFoodFound_df_light = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_lightCond/allFoodFound.npy')
allFoodFound_orl_light = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_lightCond/allFoodFound.npy')

allFoodFound_rW_dark = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/rWalk_randomFood_darkCond/allFoodFound.npy')
allFoodFound_LOrl_dark = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyORL_randomFood_darkCond/allFoodFound.npy')
allFoodFound_LDF_dark = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/expLevyDF_randomFood_darkCond/allFoodFound.npy')
allFoodFound_df_dark = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/df_randomFood_darkCond/allFoodFound.npy')
allFoodFound_orl_dark = np.load('/media/gwdg-backup/BackUp/Irene/simulationResults/orl_randomFood_darkCond/allFoodFound.npy')


import pandas as pd
import seaborn as sns
#__, pVal = stats.f_oneway(allFoodFound_rW, allFoodFound_LOrl, allFoodFound_LDF, allFoodFound_df, allFoodFound_orl)
d={'random walk light': allFoodFound_rW_light[:,0],
   'Levy ORL light':allFoodFound_LOrl_light[:,0],
   'Levy DF light': allFoodFound_LDF_light[:,0],                    
   'ORL light': allFoodFound_orl_light[:,0], 
   'dark-fly light': allFoodFound_df_light[:,0],
   'random walk dark': allFoodFound_rW_dark[:,0],
   'Levy ORL dark':allFoodFound_LOrl_dark[:,0],
   'Levy DF dark': allFoodFound_LDF_dark[:,0],                    
   'ORL dark': allFoodFound_orl_dark[:,0], 
   'dark-fly dark': allFoodFound_df_dark[:,0]}
df = pd.DataFrame.from_dict(d, orient='index')
df=df.T
#plt.figure()
sns.boxplot(data=df, notch=True)
#sns.swarmplot(data=df, edgecolor='black')
ax = plt.gca()
ax.tick_params(axis = 'both', labelsize = 20)
ax.set_ylabel('No. of foods collected', fontsize = 24) 
ax.set_title('Efficiency in finding randomly dispersed foods', fontsize = 36)