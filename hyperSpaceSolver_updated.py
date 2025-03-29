#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:23:28 2018

@author: iaji & bgeurten

line segment intersection using vectors
see Computer Graphics by F.S. Hill
"""

import numpy as np
from shapely.geometry import LineString

class hyperSpaceSolver_updated():
    
    def __init__(self,vector= np.array([[-9000.0,-9000.0],[-14000.0,-16000.0]]),border = 10000):
        # We expect the vector to be smaller than an arena size, all arenas are 
        # quadratic.
        # We have to define the four lines that can be crossed by the vector.
        self.leftLine   = np.array([[-border,-border],[-border, border]])
        self.rightLine  = np.array([[ border,-border],[ border, border]])
        self.topLine    = np.array([[-border, border],[ border, border]])
        self.bottomLine = np.array([[-border,-border],[ border,-border]])
        self.vector     = vector
        self.border     = border
        self.subSteps   = []
        
        #vector from the origin
        vecRel = self.vector[1]-self.vector[0]
        # angle of the vector
        self.theta = np.arctan2(vecRel[0],vecRel[1])
        
        self.rotMatrix = np.array([[np.cos(-1*self.theta), -np.sin(-1*self.theta)], 
                                   [np.sin(-1*self.theta),  np.cos(-1*self.theta)]])
        
        self.vecNorm = np.sqrt(vecRel[0]**2+vecRel[1]**2)
        

    def seg_intersect(self,A,B):
       
        line1 = LineString([(A[0,0],A[0,1]), (A[1,0],A[1,1])])
        line2 = LineString([(B[0,0],B[0,1]), (B[1,0],B[1,1])])
        
        result = line1.intersection(line2)
        
        if len(result.bounds) == 0:
            return (np.nan,np.nan)
        else:
            intersection = (result.coords.xy[0][0],result.coords.xy[1][0])
            if intersection[0] ==self.vector[0,0] and intersection[1] ==self.vector[0,1]:
                return (np.nan,np.nan)
            else:
                return intersection 
        
    
    def createVectorFromNormAndDirection(self,entryPoint):
        # create an passive Fick Rotation Matrix
        newVec = np.dot(self.rotMatrix, np.array([0,self.vecNorm]))
        self.vector = np.vstack((entryPoint, newVec+entryPoint))
    
    def updateVectorNorm(self):
        # calculate the norm of our substep
        vecRel = self.subSteps[-2]-self.subSteps[-1]
        #subtract substep vector norm from original vector norm
        self.vecNorm = self.vecNorm - np.sqrt(vecRel[0]**2+vecRel[1]**2)
    
    def calcAllSegs(self):
        
        #### WARNING STILL HAVE TO SOLVE SPECIAL CASE IF THE VECTOR GOES OUT 
        # THROUGH CORNER BOTH INTERSECTION COORDINATES HAVE TO CHANGE SIGN!!
        #CHECK CHANGES MADE, SOMETHING WRONG IS OCCURRING!!!!!
        intersection = self.seg_intersect(self.vector ,self.leftLine)
        if np.isnan(intersection[0])==False:
            #check if intersection occurs at the left corners (top, bottom)
            if (intersection[0]==-self.border and intersection[1]==self.border) or (intersection[0]==-self.border and intersection[1]==-self.border):
            #if np.logical_or(round(intersection[0])==-self.border,round(intersection[1])==self.border) or np.logical_or(round(intersection[0])==-self.border, round(intersection[1])==-self.border):
                entryPoint = (intersection[0]*-1, intersection[1]*-1)
            #otherwise, intersection does not occur at either left corner
            else:
                entryPoint = (intersection[0]*-1,intersection[1])

        if np.isnan(intersection[0]):
            intersection = self.seg_intersect(self.vector,self.rightLine)
            if np.isnan(intersection[0])==False:
                #check if intersection occurs at the right corners (top, bottom)
                if (intersection[0]==self.border and intersection[1]==self.border) or (intersection[0]==self.border and intersection[1]==-self.border):
                #if np.logical_or(round(intersection[0])==self.border,round(intersection[1])==self.border) or np.logical_or(round(intersection[0])==self.border, round(intersection[1])==-self.border):
                    entryPoint = (intersection[0]*-1, intersection[1]*-1)
                #otherwise, intersection does not occur at either right corner
                else:
                    entryPoint = (intersection[0]*-1,intersection[1])  
            
        if np.isnan(intersection[0]):
            intersection = self.seg_intersect(self.vector,self.topLine)
            entryPoint = (intersection[0],intersection[1]*-1)
            
        if np.isnan(intersection[0]):
            intersection = self.seg_intersect(self.vector,self.bottomLine)
            entryPoint = (intersection[0],intersection[1]*-1)

        return [intersection,entryPoint]
    
        
    def solve(self):
        
        [newInterSec,entryPoint] = self.calcAllSegs()
        self.subSteps = np.vstack((self.vector[0],newInterSec))
        self.updateVectorNorm()
        self.createVectorFromNormAndDirection(entryPoint)

        while np.isnan(newInterSec[0]) == False:
            self.subSteps = np.vstack((self.subSteps,self.vector[0]))
            [newInterSec,entryPoint] = self.calcAllSegs()
            if np.isnan(newInterSec[0]):
                self.subSteps = np.vstack((self.subSteps,self.vector[1]))
                
            else:
                self.subSteps = np.vstack((self.subSteps,newInterSec))
                self.updateVectorNorm()
                self.createVectorFromNormAndDirection(entryPoint)
                
        
        return self.subSteps




