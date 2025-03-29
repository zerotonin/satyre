#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:33:09 2018

@author: iaji
"""

import numpy as np 
import pandas as pd
from scipy.spatial import Delaunay


class walkSim_darkFly:
    """Simulation of real fly's exploratory strategies based on experimental data and Markov model in condition with light"""
    def __init__(self,targetPM_cumsum_probabilityMatrix, PM_index, VELO_arr, nrTrial = 1000,
                border = 10000.0, maxSteps = 1000,totalFood = 100,foodMode='random'):
        self.border              = border
        self.bodyWidth           = 2
        self.bodyLength          = 3
        self.bodyLengthSq        = self.bodyLength**2
        self.bodyLengthVec       = np.array([[0, self.bodyLength]])
        self.targetPM_cumsum_probabilityMatrix = targetPM_cumsum_probabilityMatrix
        self.PM_index            = PM_index
        self.VELO_arr            = VELO_arr
        self.simulationResolution = 500 #point per second
        self.startPoint          = np.array([[0,0]])
        self.nrTrial             = nrTrial
        self.count               = 0
        self.stepCounter         = 0
        self.currentPos          = self.startPoint
        self.endPos              = self.startPoint
        self.maxSteps            = maxSteps
        self.foodMode            = foodMode
        self.totalFood           = totalFood
        self.hyperspaceCount    = 0
        self.startListX          = []
        self.startListX.append(self.startPoint[0,0])
        self.startListY          = []
        self.startListY.append(self.startPoint[0,1])
        self.endListX            = []
        self.endListY            = []
        self.foodFound_coordList = []
        self.stepCounts          = []
        self.FirstPM = self.findFirstPM()
        self.all_performedPMs = []
        self.all_performedPMs.append(self.FirstPM)
        self.bodyAngles_list = []
        self.bodyAngles_list.append(0)
        self.headCoords_list = []
        self.headCoords_list.append(self.bodyLengthVec/2)
        self.tailCoords_list = []
        self.tailCoords_list.append(self.bodyLengthVec/2*-1)
        self.midCoords_list = []
        self.cumYaw = 0
    
    def findFirstPM(self):
        medianVelocities = np.nanmedian(self.VELO_arr, axis=1)
        self.FirstPM = np.nanargmin(abs(medianVelocities))
        return self.FirstPM

    def scatterFood(self):
        if self.foodMode == 'random':
            self.scatterFood_random()
        elif self.foodMode == 'clustered':
            self.scatterFood_cluster()
    
    def scatterFood_random(self):
        self.foodBorder = 60
        self.foodPos=np.random.randint(-self.foodBorder,self.foodBorder, size=(self.totalFood,2))
        
    def scatterFood_cluster(self):
        self.foodBorder = 60
        no_food_perCluster = 10
        no_cluster = int(self.totalFood/no_food_perCluster)
        self.foodPos_center=np.random.randint(-self.foodBorder,self.foodBorder, size=(no_cluster,2))
        foodPos = []
        for i in range(len(self.foodPos_center)):
            foodPosX = np.random.normal(self.foodPos_center[i,0], scale=1, size=(no_food_perCluster))
            foodPosY = np.random.normal(self.foodPos_center[i,1], scale=1, size=(no_food_perCluster))
            for i in range(len(foodPosX)):
                foodPos.append((foodPosX[i], foodPosY[i]))
        self.foodPos = np.array(foodPos)
                
    
    def throwDice(self):
        diceNumber = np.random.uniform(0,1,1)
        return diceNumber

    def hyperspace(self):
        from hyperSpaceSolver_updated import hyperSpaceSolver_updated
        self.hsSolver=hyperSpaceSolver_updated(vector=np.vstack((self.currentPos, self.endPos)), border=self.border)
        return self.hsSolver.solve()
    
    def in_hull(self):
        """Check if there are foods within the fly's polygonal mechanosensory field as it walks from one point to another."""
        flyPos2 = np.array([[self.startListX[-1], self.startListY[-1]]])        
        distance2 = self.foodPos-flyPos2
        distance = np.sum(distance2**2,axis=1) <= self.bodyLengthSq #and np.sqrt(np.sum(distance1**2,axis=1)) <= self.bodyLength*2
        p = self.foodPos[distance]
        points = self.polygonPoints        
        if not isinstance(points,Delaunay) and len(p) > 0:
            points = Delaunay(points)
            inpoly =  points.find_simplex(p)>=0
            distance[distance] = inpoly
        return distance

    def foodroutine(self):
        """ 
        foodroutine () checks if one or more foods in foodPos array lies within the fly's trapezoid mechanosensory field, 
            removes the found food from the foodPos array, and returns the updated foodPos as well as its new length
        input: two coordinates p1 and p2 (both are 1x2 arrays), an array of coordinates of food positions (foodPos), and the agent's body width
        output: new updated foodPos and its length"""
        self.polygonPoints = np.array([self.tailCoords_list[-2][0], self.headCoords_list[-2][0], self.tailCoords_list[-1][0], self.headCoords_list[-1][0]])
        foodFound_boolIDX = self.in_hull()
#        foodFoundIDX = np.nonzero(foodFound_boolIDX)
        foodFoundCoord = self.foodPos[foodFound_boolIDX]
        self.foodPos   = self.foodPos[~foodFound_boolIDX] #  np.delete(self.foodPos, (foodFoundIDX ), axis=0)
        self.totalFood_updated = len(self.foodPos)
        return foodFoundCoord
    
    def calcEndOfSightPosition(self):
        slopeHeight = (self.endPos[0,1] - self.currentPos[0,1]) / (self.endPos[0,0] - self.currentPos[0,0]) 
        sightLimit = 10*self.bodyLength
        theta = np.arctan(slopeHeight)
        xEnd = sightLimit*np.cos(theta) + self.currentPos[0,0]
        yEnd = sightLimit*np.sin(theta) + self.currentPos[0,1]
        self.endPos = np.array([[xEnd, yEnd]])
        
    def calcRectangleCorners(self):
        lengthWidth = self.bodyWidth
        slopeHeight = (self.endPos[0,1] - self.currentPos[0,1]) / (self.endPos[0,0] - self.currentPos[0,0])
        slopeWidth = -1/slopeHeight
        dx = np.sqrt(lengthWidth**2/(1+slopeWidth**2))/2
        dy = slopeWidth * dx
        self.polygonPoints = np.ones((4,2))*np.nan
        self.polygonPoints[0,:] = np.array([self.currentPos[0,0]+dx, self.currentPos[0,1]+dy])
        self.polygonPoints[1,:] = np.array([self.currentPos[0,0]-dx, self.currentPos[0,1]-dy])
        self.polygonPoints[2,:] = np.array([self.endPos[0,0]+dx, self.endPos[0,1]+dy])
        self.polygonPoints[3,:] = np.array([self.endPos[0,0]-dx, self.endPos[0,1]-dy])
        
    def foodroutine_light(self):
        self.calcEndOfSightPosition()
        self.calcRectangleCorners()
        foodFound_boolIDX = self.in_hull()
        foodFoundCoord = self.foodPos[foodFound_boolIDX]
        self.foodPos   = self.foodPos[~foodFound_boolIDX] #  np.delete(self.foodPos, (foodFoundIDX ), axis=0)
        self.totalFood_updated = len(self.foodPos)
        return foodFoundCoord
        
    def foodroutine_hyperspace(self):
        self.calcRectangleCorners()
        foodFound_boolIDX = self.in_hull()
        foodFoundCoord = self.foodPos[foodFound_boolIDX]
        self.foodPos   = self.foodPos[~foodFound_boolIDX] #  np.delete(self.foodPos, (foodFoundIDX ), axis=0)
        self.totalFood_updated = len(self.foodPos)
        return foodFoundCoord
          
    def calcNewPosition(self):
        theta = self.bodyAngle + self.trajectoryAngle
        rotMatrix = np.array([[np.cos(-1*theta), -np.sin(-1*theta)], 
                                       [np.sin(-1*theta),  np.cos(-1*theta)]]) #-1*theta --> clockwise rotation
        dispVec = np.array([[self.slipDistance, self.thrustDistance]])
        rotatedVec = np.dot(rotMatrix, dispVec.T)
        self.endPos = self.currentPos + rotatedVec.T
        return self.endPos
    
    def calcBodyPositions(self):
        theta = self.bodyAngle + self.trajectoryAngle
        rotMatrix = np.array([[np.cos(-1*theta), -np.sin(-1*theta)], 
                                       [np.sin(-1*theta),  np.cos(-1*theta)]]) #-1*theta --> clockwise rotation
        rotatedVec = np.dot(rotMatrix, np.array([[0, self.stepDistance]]).T)
        self.endPos = self.currentPos + rotatedVec.T
        return self.endPos
              
    def sim_levy(self):
        """Throw dice, get the protoypical movement based on current prototypical movement
        ---> Markov model at use"""
        diceNumber = self.throwDice()
        if diceNumber < 1:
            targetPM_idx=min(np.where(self.targetPM_cumsum_probabilityMatrix[self.all_performedPMs[-1],:] > diceNumber)[0])
        elif diceNumber ==1:
            targetPM_idx=min(np.where(self.targetPM_cumsum_probabilityMatrix[self.all_performedPMs[-1],:] == diceNumber)[0])
        self.targetPM = int(self.PM_index[self.all_performedPMs[-1],targetPM_idx])
        self.all_performedPMs.append(self.targetPM)
        """Calculate step distance and direction of movement based on velocities of the chosen protoypical movement"""
        velocities_targetPM = self.VELO_arr[self.targetPM]
        self.thrustDistance = velocities_targetPM[0]/self.simulationResolution
        self.slipDistance = velocities_targetPM[1]/self.simulationResolution
        self.trajectoryAngle = velocities_targetPM[2]/self.simulationResolution
        """Calculate the coordinate of the next position based on step distance and direction of movement"""
        self.cumYaw += self.trajectoryAngle
        self.bodyAngle  = self.cumYaw
        self.bodyAngles_list.append(self.trajectoryAngle)
        self.currentPos = self.startPoint        
        self.startPoint = self.calcNewPosition()
        self.stepCounter += 1
        if self.startPoint[0,0] > self.border or self.startPoint[0,0] < -self.border or self.startPoint[0,1] > self.border or self.startPoint[0,1] < -self.border:
           self.hyperspaceCount += 1
           self.currentPos = np.array([[self.coordinatesX[-1], self.coordinatesY[-1]]])
           self.endPos = np.array([[self.startPoint[0,0], self.startPoint[0,1]]])
           subSteps = self.hyperspace() 
           starts = subSteps[::2]
           stops = subSteps[1::2]  
           self.stepCounts += len(stops)*[self.stepCounter] 
           ##Storing the substeps into the lists
           ##note: starts starts from the second substep because 1st substep is already appended as the startPoint
           for i in range(1,len(starts)): 
               self.startListX.append(starts[i,0])
               self.startListY.append(starts[i,1])
           for i in range(len(stops)):
               self.endListX.append(stops[i,0])
               self.endListY.append(stops[i,1])
           for i in range(1,len(subSteps)):
               self.allStepsX.append(subSteps[i,0])
               self.allStepsY.append(subSteps[i,1])
           #Food routine
           for i in range(len(starts)):
               self.currentPos = np.array([[starts[i,0], starts[i,1]]])
               self.endPos= np.array([[stops[i,0], stops[i,1]]])
               foodFoundCoord = self.foodroutine_hyperspace()
               self.foodFound_coordList.append(foodFoundCoord)    
           self.startPoint = np.array([[stops[-1,0], stops[-1,1]]])
           self.startListX.append(stops[-1,0])
           self.startListY.append(stops[-1,1])

        else:
            self.endListX.append(self.startPoint[0,0])
            self.endListY.append(self.startPoint[0,1])
            self.startListX.append(self.startPoint[0,0])
            self.startListY.append(self.startPoint[0,1])
            self.stepCounts.append(self.stepCounter)
            """Food routine"""
            """if protoypical movement allows optic flow, fly sees food --> do the food routine in light.
            Otherwise fly can't see food and collects food only if it's within its mechanosensory field"""
            if velocities_targetPM[0] >= 0.1 and velocities_targetPM[2] < np.deg2rad(30):
                self.currentPos = np.array([[self.startListX[-2], self.startListY[-2]]])
                self.endPos= np.array([[self.startListX[-1], self.startListY[-1]]]) 
                foodFoundCoord = self.foodroutine_light()
                self.foodFound_coordList.append(foodFoundCoord)
            else:
                """Calculate positions of body parts: mid, tail, head"""
                self.stepDistance = self.bodyLength/2
                self.currentPos = np.array([[self.startListX[-1], self.startListY[-1]]])
                self.midCoords_list.append(self.currentPos)
                self.bodyAngle = (self.cumYaw-self.trajectoryAngle)-np.deg2rad(180)
                self.trajectoryAngle = 0
                self.tailCoord = self.calcBodyPositions()
                self.tailCoords_list.append(self.tailCoord)
                self.bodyAngle = (self.cumYaw-self.trajectoryAngle)
                self.headCoord = self.calcBodyPositions()
                self.headCoords_list.append(self.headCoord)
                """Food routine"""
                foodFoundCoord = self.foodroutine()
                self.foodFound_coordList.append(foodFoundCoord)
        self.count +=1
        
            
    def clearObject(self,startPoint =np.array([[0,0]])):
        self.startPoint    = startPoint
        self.startListX    = []
        self.startListX.append(self.startPoint[0,0])
        self.startListY    = []
        self.startListY.append(self.startPoint[0,1])
        self.endListX      = []
        self.endListY      = []
        self.stepCounts    = []
        self.foodFound_coordList = []
        self.countTrial    = 0
        self.count         = 0
        self.stepCounter   = 0
        self.currentPos    = self.startPoint
        self.endPos        = self.startPoint
        self.all_performedPMs = []
        self.all_performedPMs.append(self.FirstPM)
        self.bodyAngles_list = []
        self.bodyAngles_list.append(0)
        self.headCoords_list = []
        self.headCoords_list.append(np.array([[0, self.bodyLength/2]]))
        self.tailCoords_list = []
        self.tailCoords_list.append(np.array([[0,-self.bodyLength/2]]))
        self.midCoords_list = []
        

    def simulate(self):
        self.clearObject(startPoint =np.array([[0,0]]))
        self.scatterFood()
        while self.count <= self.maxSteps:
            self.sim_levy()
        self.foodFound_total = self.totalFood - self.totalFood_updated
    
    def simulateMultiple(self):
        self.foodFound_total_allTrials = np.ones((self.nrTrial, 1))*np.nan
        c = 0
        while c < self.nrTrial:
            self.simulate()
            c +=1
            print (c)
            self.foodFound_total_allTrials[c-1,0] = self.foodFound_total
            
    def saveAllData(self):        
        self.startListXY_arr = np.zeros((len(self.startListX),2))
        self.startListXY_arr[:,0]=self.startListX[:]
        self.startListXY_arr[:,1]=self.startListY[:]
        
        self.endListXY_arr = np.zeros((len(self.endListX),2))
        self.endListXY_arr[:,0]=self.endListX[:]
        self.endListXY_arr[:,1]=self.endListY[:]
        
        self.stepCounts_arr=np.zeros((len(self.stepCounts),1))
        self.stepCounts_arr[:,0]=self.stepCounts[:]

        print (len(self.stepCounts_arr))
        print (len(self.startListXY_arr))
        print (len(self.endListXY_arr))
        keys = []
        for i in range(len(self.stepCounts)):
            keys.append(str(i))
        data={}
        for i in range(len(self.endListXY_arr)):
            data[keys[i]] = [self.stepCounts[i], self.startListXY_arr[i,0], self.startListXY_arr[i,1], 
                    self.endListXY_arr[i,0], self.endListXY_arr[i,1], self.foodFound_coordList[i]] 
        self.dataframe = pd.DataFrame.from_dict(data, orient='index', columns=['step number', 'start X', 'start Y', 'end X', 
                                                                                'end Y', 'position of food found'])
