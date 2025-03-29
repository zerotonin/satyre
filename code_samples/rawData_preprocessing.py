#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:39:35 2018

@author: iaji
"""
"""Input raw data are .txt files that contain:
    1. IDX.txt = a sequence of prototypical movements (PMs) recorded in experiment (2000 PMs in total)
    2. C.txt = the velocities (forward, sideward, rotational) of each protoypical movement
    
    The output are matrices of size 2000x2000, where :
        1. targetPM_cumsum_probabilityMatrix: contains cumulative probability matrix
        2. PM_index: index of rows = starting PM and each value in a cell = target PM"""

import numpy as np

"""Import txt. file""" #use np.loadtxt next time 
text_file = open("/media/gwdg-backup/BackUp/Irene/ClustResult/OR/ORL_IDX.txt", "r")
lines = text_file.readlines()
IDX_arr= np.zeros((len(lines)))
for i in range(len(lines)):
    IDX_arr[i] = lines[i]
    
c_file = open("/media/gwdg-backup/BackUp/Irene/ClustResult/OR/ORL_C.txt", "r")
c_lines = c_file.readlines()
c_list = [int(x) for x in '3 ,2 ,6 '.split(',')]
VELO_arr = np.ones((len(c_lines)+1,3))*np.nan
for i in range(len(c_lines)):
    PM_velo = [float(x) for x in c_lines[i].split(',')]
    VELO_arr[i+1,:] = PM_velo

"""Initialize empty matrix"""
targetPM_frequencyMatrix = np.ones((int(max(IDX_arr)+1), int(max(IDX_arr)+1)))*np.nan
for i in range(1, int(max(IDX_arr)+1)):
    targetPM_frequencyMatrix[i,0] = i
    targetPM_frequencyMatrix[0,i] = i
 
def get_followingNumber(IDX_arr, number_of_interest):
    """get_followingNumber() receives an array of prototypical movements (PMs) sequence and a PM number of interest
    and returns a list of PM numbers that occur after the input PM number"""
    allNumbers = []
    for i in range(len(IDX_arr)):
        if IDX_arr[i] == number_of_interest:
            #if iterator reaches the end of the list, return nan
            if i == len(IDX_arr)-1:
                allNumbers.append(np.nan)
            else:
                allNumbers.append(IDX_arr[i+1])
    return allNumbers

"""lists_of_followingPMs is a list that contains lists, each of which contains all the PM 
that follow a certain PM (indicated by the index of list in lists_of_followingPMs +1)"""
lists_of_followingPMs = []
for i in range(1, int(max(IDX_arr)+1)):
    lists_of_followingPMs.append(get_followingNumber(IDX_arr, i))
    print(i)     

"""fill the empty targetPM_frequencyMatrix
targetPM_frequencyMatrix: rows --> start PMs
                          columns --> target PMs
each point represents the frequency of a target PM occurring after a start PM"""
for startPM_idx in range(0, int(max(IDX_arr))):
    for targetPM in range(1, int(max(IDX_arr)+1)):
        targetPM_frequencyMatrix[startPM_idx+1, targetPM] = lists_of_followingPMs[startPM_idx].count(targetPM)

"""Convert absolute frequency to probability:
    Calculate the probability of a PM occuring after a certain PM"""
targetPM_probabilityMatrix = np.ones((int(max(IDX_arr)+1), int(max(IDX_arr)+1)))*np.nan
for i in range(1, int(max(IDX_arr)+1)):
    targetPM_probabilities = targetPM_frequencyMatrix[i,1:]/sum(targetPM_frequencyMatrix[i,1:])
    targetPM_probabilityMatrix[i,1:] = targetPM_probabilities

"""Sort the probabilities from lowest to highest (to prepare for cumulative sum probability) 
while keeping the original index"""
sortedProbabilityMatrix = np.ones((int(max(IDX_arr)+1), int(max(IDX_arr)+1)))*np.nan
PM_index = np.ones((int(max(IDX_arr)+1), int(max(IDX_arr)+1)))*np.nan
for i in range(1, int(max(IDX_arr)+1)):
    sortedProbabilityMatrix[i,1:] = np.sort(targetPM_probabilityMatrix[i,1:])
    PM_index[i,1:] = np.argsort(targetPM_probabilityMatrix[i,1:])+1

"""Calculate the cumulative sum probabilities"""
targetPM_cumsum_probabilityMatrix = np.ones((int(max(IDX_arr)+1), int(max(IDX_arr)+1)))*np.nan
for i in range(1, int(max(IDX_arr)+1)):
    targetPM_cumsum_probabilityMatrix[i,1:] = np.cumsum(sortedProbabilityMatrix[i,1:])


