#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:33:09 2018

@author: iaji
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class walkSim:
    """Simulation of Levy flight in a virtual arena, with a virtual fly and virtual foods scattered across the arena.
    self.foodFound_total shows how many foods were collected at the end of simulation.
    ws.dataframe shows all positions (in x,y-coordinates) of the fly and the position of the foods that were collected
    ws.plotTrajectories plots the trajectories of the fly during the simulation."""
    def __init__(self,cauchyAlpha =1, nrTrial = 1000,
                border = 10000.0, mode='cauchy',maxSteps = 1000,totalFood = 10000,foodMode='random'):

        #the higher alpha is, the closer -1/alpha converges to zero --> f=1
        self.alpha               = cauchyAlpha
        self.border              = border
        self.bodyWidth           = 2
        self.bodyLength          = 7
        self.startPoint          = np.array([[0,0]])
        self.startListX          = []
        self.startListX.append(self.startPoint[0,0])
        self.startListY          = []
        self.startListY.append(self.startPoint[0,1])
        self.endListX            = []
        self.endListY            = []
        self.foodFound_coordList = []
        self.stepCounts          = []
        self.nrTrial             = nrTrial
        self.count               = 0
        self.stepCounter         = 0
        self.currentPos          = self.startPoint
        self.endPos              = self.startPoint
        self.mode                = mode
        self.maxSteps           = maxSteps
        self.foodMode           = foodMode
        self.totalFood          = totalFood
        self.scatterFood()
        self.hyperspaceCount    = 0
        self.bodyAngles          = []
        self.bodyAngles.append(0)
           
    def scatterFood(self):
        if self.foodMode == 'random':
            self.scatterFood_random()
        elif self.foodMode == 'clustered':
            self.scatterFood_cluster()
    
    def scatterFood_random(self):
        self.foodBorder = 10000
        self.foodPos=np.random.randint(-self.foodBorder,self.foodBorder, size=(self.totalFood,2))
        
    def scatterFood_cluster(self):
        self.foodBorder = 10000
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
    
    def initUniform(self):
            self.stepSizeFunc = lambda : np.random.normal()
            return self.stepSizeFunc
            
    def initCauchy(self):
            self.stepSizeFunc = lambda: np.random.uniform()**(-1/self.alpha)
            return self.stepSizeFunc

    def hyperspace(self):
        from hyperSpaceSolver_updated import hyperSpaceSolver_updated
        self.hsSolver=hyperSpaceSolver_updated(vector=np.vstack((self.currentPos, self.endPos)), border=self.border)
        return self.hsSolver.solve()
    
    def in_hull(self):
        """Check if there are foods within the fly's polygonal mechanosensory field as it walks from one point to another."""
        flyPos2 = np.array([[self.startListX[-1], self.startListY[-1]]])        
        distance2 = self.foodPos-flyPos2
        distance = np.sum(distance2**2,axis=1) <= self.bodyLength**2 
        p = self.foodPos[distance]
        points = self.polygonPoints        
        if not isinstance(points,Delaunay) and len(p) > 0:
            points = Delaunay(points)
            inpoly =  points.find_simplex(p)>=0
            distance[distance] = inpoly
        return distance
        
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
                
    def foodroutine(self):
        """ 
        foodroutine () checks if one or more foods in foodPos array lies within the fly's rectangular mechanosensory field (bodyWidth x stepDistance), 
            removes the found food from the foodPos array, and returns the updated foodPos as well as its new length
        input: two coordinates p1 and p2 (both are 1x2 arrays), an array of coordinates of food positions (foodPos), and the agent's body width
        output: new updated foodPos and its length"""
        self.calcRectangleCorners()
        foodFound_boolIDX = self.in_hull()
        foodFoundCoord = self.foodPos[foodFound_boolIDX]
        self.foodPos   = self.foodPos[~foodFound_boolIDX] #  np.delete(self.foodPos, (foodFoundIDX ), axis=0)
        self.totalFood_updated = len(self.foodPos)
        return foodFoundCoord
       
    def FickTransform(self):
        theta = self.bodyAngle + self.trajectoryAngle
        rotMatrix = np.array([[np.cos(-1*theta), -np.sin(-1*theta)], 
                                       [np.sin(-1*theta),  np.cos(-1*theta)]]) #-1*theta --> clockwise rotation
        rotatedVec = np.dot(rotMatrix, np.array([[0, self.stepDistance]]).T)
        self.startPoint = self.startPoint + rotatedVec.T
        return self.startPoint       
       
    def sim_levy(self):
        """"Select mode"""
        if self.mode == 'cauchy':
            self.stepSizeFunc = self.initCauchy()
        elif self.mode == 'uniform':
            self.stepSizeFunc = self.initUniform()
        """Choose angular direction randomly and calculate step size based on selected mode"""
        self.directionFunc = lambda: np.random.uniform()*2*np.pi
        angle = self.directionFunc()
        step  = self.stepSizeFunc()
        self.bodyAngles.append(angle)
        self.startPoint = np.array([[self.startPoint[0,0]+step*np.cos(angle), self.startPoint[0,1]+step*np.sin(angle)]])
        self.stepCounter += 1
        """Do hyperspace routine if fly exits the virtual arena --> fly re-enters from opposite side"""
        if self.startPoint[0,0] > self.border or self.startPoint[0,0] < -self.border or self.startPoint[0,1] > self.border or self.startPoint[0,1] < -self.border:
           print('hyperspace')
           self.hyperspaceCount += 1
           print (self.stepCounter)
           self.currentPos = np.array([[self.startListX[-1], self.startListY[-1]]])
           self.endPos = np.array([[self.startPoint[0,0], self.startPoint[0,1]]])
           subSteps = self.hyperspace() 
           starts = subSteps[::2]
           stops = subSteps[1::2]  
           self.stepCounts += len(stops)*[self.stepCounter] 
           """Storing the substeps into the lists
           ##note: starts starts from the second substep because 1st substep is already appended as the startPoint"""
           for i in range(1,len(starts)): 
               self.startListX.append(starts[i,0])
               self.startListY.append(starts[i,1])
           for i in range(len(stops)):
               self.endListX.append(stops[i,0])
               self.endListY.append(stops[i,1])
           """Collect food"""
           for i in range(len(starts)):
               self.currentPos = np.array([[starts[i,0], starts[i,1]]])
               self.endPos= np.array([[stops[i,0], stops[i,1]]])
               foodFoundCoord = self.foodroutine()
               self.foodFound_coordList.append(foodFoundCoord)    
           self.startPoint = np.array([[stops[-1,0], stops[-1,1]]])
           self.startListX.append(stops[-1,0])
           self.startListY.append(stops[-1,1])
        else:
            """"Save coordinates in lists"""
            self.endListX.append(self.startPoint[0,0])
            self.endListY.append(self.startPoint[0,1])
            self.startListX.append(self.startPoint[0,0])
            self.startListY.append(self.startPoint[0,1])
            self.stepCounts.append(self.stepCounter)
            """Collect food"""
            self.currentPos = np.array([[self.startListX[-2], self.startListY[-2]]])
            self.endPos= np.array([[self.startListX[-1], self.startListY[-1]]]) 
            foodFoundCoord = self.foodroutine()
            self.foodFound_coordList.append(foodFoundCoord)
        self.count +=1
    
    def calcStepSize(self):
        self.stepSize_arr = np.zeros([len(self.startListX)])
        for i in np.arange(1,len(self.startListX)):
            stepSize = np.sqrt((self.startListX[i]-self.startListX[i-1])**2 + (self.startListY[i]-self.startListY[i-1])**2)
            self.stepSize_arr[i]=stepSize
        return self.stepSize_arr     
    
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
        self.bodyAngles          = []
        self.bodyAngles.append(0)
        
    def simulate(self):
        self.clearObject(startPoint =np.array([[0,0]]))
        self.scatterFood()
        while self.count <= self.maxSteps:
            self.sim_levy()
        self.foodFound_total = self.totalFood - self.totalFood_updated
        self.calcStepSize()
    
    def simulateMultiple(self):
        self.foodFound_total_allTrials = np.ones((self.nrTrial, 1))*np.nan
        self.stepSize_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        c = 0
        while c < self.nrTrial:
            self.simulate()
            c +=1
            print (c)
            self.foodFound_total_allTrials[c-1,0] = self.foodFound_total
            self.stepSize_allTrials[c-1,:] = self.stepSize_arr
            
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
        self.dataframe = pd.DataFrame.from_dict(data, orient='index', columns=['step number', 
                                                                               'start X', 
                                                                               'start Y', 
                                                                               'end X', 
                                                                                'end Y', 
                                                                                'position of food found'])
            
    def plotTrajectories(self):
        plt.plot(self.startListXY_arr[:,0], self.startListXY_arr[:,1], 'blue')
        plt.axis('equal')

    def plotFood(self):
        plt.scatter(self.foodPos[:,0], self.foodPos[:,1], color='green', marker='*')
        plt.plot(self.startListXY_arr[:,0], self.startListXY_arr[:,1], 'blue')
        plt.axis('equal')

