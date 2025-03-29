#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:33:09 2018

@author: iaji
"""

import numpy as np 
from scipy.spatial import Delaunay


class walkSim_darkFly:
    """Simulation of real fly's exploratory strategies based on experimental data modeled with Markov model
    in dark condition."""
    def __init__(self,targetPM_cumsum_probabilityMatrix, PM_index, VELO_arr, nrTrial = 1000,
                border = 10000.0, maxSteps = 10000, totalFood = 1000,foodMode='random'):
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
        self.lolliPopRate        = 25
        self.hyperspaceCount    = 0
        self.coordinatesX        = []
        self.coordinatesX.append(self.startPoint[0,0])
        self.coordinatesY        = []
        self.coordinatesY.append(self.startPoint[0,1])
        self.allStepsX           = []
        self.allStepsX.append(self.startPoint[0,0])
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
        self.headCoordX_list = []
        self.headCoordX_list.append((self.bodyLengthVec/2)[0,0])
        self.headCoordY_list = []
        self.headCoordY_list.append((self.bodyLengthVec/2)[0,1])
        self.tailCoordX_list = []
        self.tailCoordX_list.append((self.bodyLengthVec/2*-1)[0,0])
        self.tailCoordY_list = []
        self.tailCoordY_list.append((self.bodyLengthVec/2*-1)[0,1])
        self.midCoords_list = []
        self.psiAngles_list = []
        self.cumYaw = 0
    
    def findFirstPM(self):
        """Agent starts at a still (no movement, low velocity) prototypical movement. 
        The function finds PM with the lowest overall velocities"""
        medianVelocities = np.nanmedian(self.VELO_arr, axis=1)
        self.FirstPM = np.nanargmin(abs(medianVelocities))
        return self.FirstPM

    def scatterFood(self):
        """Initiate food scattering in different modes: random or clustered"""
        if self.foodMode == 'random':
            self.scatterFood_random()
        elif self.foodMode == 'clustered':
            self.scatterFood_cluster()
    
    def scatterFood_random(self):
        """Scatter foods randomly. Coordinates of foods are drawn randomly"""
        self.foodBorder = 60
        self.foodPos=np.random.randint(-self.foodBorder,self.foodBorder, size=(self.totalFood,2))
        
    def scatterFood_cluster(self):
        """Scatter foods in clusters. The centric coordinate of each food cluster is drawn randomly. 
        The surrounding food coordinates are drawn from normal Gaussian distribution, with the centric coordinate being the mean"""
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
        """Get random number ranged 0-1 for picking a prototypical movement based on Markov model"""
        diceNumber = np.random.uniform(0,1,1)
        return diceNumber

    def hyperspace(self):
        """If the agent crosses the border of the virtual arena, the hyperspace routine brings the fly back through the opposite border"""
        from hyperSpaceSolver_updated import hyperSpaceSolver_updated
        self.hsSolver=hyperSpaceSolver_updated(vector=np.vstack((self.currentPos, self.endPos)), border=self.border)
        return self.hsSolver.solve()
    
    def in_hull(self):
        """Check if there are foods within the fly's polygonal mechanosensory field as it walks from one point to another."""
        flyPos2 = np.array([[self.startListX[-1], self.startListY[-1]]])        
        distance2 = self.foodPos-flyPos2
        distance = np.sum(distance2**2,axis=1) <= self.bodyLengthSq #and np.sqrt(np.sum(distance1**2,axis=1)) <= self.bodyLength*2
        p = self.foodPos[distance]
        #print(len(p))
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
        foodroutine () checks if one or more foods in foodPos array lies within the fly's trapezoid mechanosensory field, 
            removes the found food from the foodPos array, and returns the updated foodPos as well as its new length
        input: two coordinates p1 and p2 (both are 1x2 arrays), an array of coordinates of food positions (foodPos), and the agent's body width
        output: new updated foodPos and its length"""
        self.polygonPoints = np.array([[self.tailCoordX_list[-2], self.tailCoordY_list[-2]], 
                                       [self.headCoordX_list[-2], self.headCoordY_list[-2]],
                                       [self.tailCoordX_list[-1], self.tailCoordX_list[-1]], 
                                       [self.tailCoordX_list[-1], self.tailCoordX_list[-1]]])        
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
        """calcNewPosition() calculates the coordinates of the next step.
        The x-component of the step distance (dispVec) is equal to slipDistance (displacement due to slip/sideward velocity).
        The y-component of the step distance (dispVec) is equal to thrustDistance (displacement due to thrust/forward-backward velocity).
        dispVec is rotated by the yaw angle (angular displacement due to yaw/rotational velocity) to give the correct orientation of the
        trajectory"""
        theta = self.bodyAngle + self.trajectoryAngle
        rotMatrix = np.array([[np.cos(-1*theta), -np.sin(-1*theta)], 
                                       [np.sin(-1*theta),  np.cos(-1*theta)]]) #-1*theta --> clockwise rotation
        dispVec = np.array([[self.slipDistance, self.thrustDistance]])
        rotatedVec = np.dot(rotMatrix, dispVec.T)
        self.endPos = self.currentPos + rotatedVec.T
        return self.endPos
    
    def calcBodyPositions(self):
        """calcBodyPositions() is used to calculate the coordinates of the head and tail of the fly,
        where stepDistance = fly's body length/2 (mid body part-head or mid body part-tail)
        and currentPos = current position of the fly's mid body part"""
        theta = self.bodyAngle + self.trajectoryAngle
        rotMatrix = np.array([[np.cos(-1*theta), -np.sin(-1*theta)], 
                                       [np.sin(-1*theta),  np.cos(-1*theta)]]) #-1*theta --> clockwise rotation
        rotatedVec = np.dot(rotMatrix, np.array([[0, self.stepDistance]]).T)
        self.endPos = self.currentPos + rotatedVec.T
        return self.endPos
    
    def calcPsiAngle(self):
        # Drift angle (psi angle) is the angle between the heading (orientation of the body) and the trajectory (track)
        vecBodyHeading = (self.headCoord[0,0]-self.tailCoord[0,0], self.headCoord[0,1]-self.tailCoord[0,1])
        vecTrajectory = (self.startListX[-1]-self.startListX[-2], self.startListY[-1]-self.startListY[-2])
        self.psi = np.arccos(np.dot(vecBodyHeading,vecTrajectory)/(np.linalg.norm(vecBodyHeading)*np.linalg.norm(vecTrajectory)))
        return self.psi
    
              
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
        self.cumYaw += self.trajectoryAngle
        self.bodyAngle  = self.cumYaw
        self.bodyAngles_list.append(self.trajectoryAngle)
        self.currentPos = self.startPoint
        """Calculate the coordinate of the next position based on step distance and direction of movement"""
        self.startPoint = self.calcNewPosition()
        self.stepCounter += 1
        """Hyperspace routine only if next position's coordinate is outside of the virtual arena"""
        if self.startPoint[0,0] > self.border or self.startPoint[0,0] < -self.border or self.startPoint[0,1] > self.border or self.startPoint[0,1] < -self.border:
           print('hyperspace')
           self.hyperspaceCount += 1
           print (self.stepCounter)
           """Fly exits from one side of the virtual arena and re-enters at the opposite side.
           Calculate the new coordinate after re-entry"""
           self.currentPos = np.array([[self.coordinatesX[-1], self.coordinatesY[-1]]])
           self.endPos = np.array([[self.startPoint[0,0], self.startPoint[0,1]]])
           subSteps = self.hyperspace() 
           starts = subSteps[::2]
           stops = subSteps[1::2]  
           self.stepCounts += len(stops)*[self.stepCounter] #stepCounter is repeated for the substeps, as the substeps made up the whole step
           """Storing the substeps into the lists
           note: starts starts from the second substep because 1st substep is already appended as the startPoint"""
           for i in range(1,len(starts)): 
               self.startListX.append(starts[i,0])
               self.startListY.append(starts[i,1])
           for i in range(len(stops)):
               self.endListX.append(stops[i,0])
               self.endListY.append(stops[i,1])
           for i in range(1,len(subSteps)):
               self.allStepsX.append(subSteps[i,0])
               self.allStepsY.append(subSteps[i,1])
           """Do the food routine. Check if fly passes foods in each of its substeps. If yes-->food collected"""
           for i in range(len(starts)):
               self.currentPos = np.array([[starts[i,0], starts[i,1]]])
               self.endPos= np.array([[stops[i,0], stops[i,1]]])
               foodFoundCoord = self.foodroutine_hyperspace()
               self.foodFound_coordList.append(foodFoundCoord)    
           """New position is now the current position"""
           self.startPoint = np.array([[stops[-1,0], stops[-1,1]]])
           self.startListX.append(stops[-1,0])
           self.startListY.append(stops[-1,1])
        else:
            self.endListX.append(self.startPoint[0,0])
            self.endListY.append(self.startPoint[0,1])
            self.startListX.append(self.startPoint[0,0])
            self.startListY.append(self.startPoint[0,1])
            self.stepCounts.append(self.stepCounter)
            """Calculate coordinates of body"""
            self.stepDistance = self.bodyLength/2
            self.currentPos = np.array([[self.startListX[-1], self.startListY[-1]]])
            self.midCoords_list.append(self.currentPos)
            self.bodyAngle = (self.cumYaw-self.trajectoryAngle)-np.deg2rad(180)
            self.trajectoryAngle = 0
            self.tailCoord = self.calcBodyPositions()
            self.tailCoordX_list.append(self.tailCoord[0,0])
            self.tailCoordY_list.append(self.tailCoord[0,1])
            self.bodyAngle = (self.cumYaw-self.trajectoryAngle)
            self.headCoord = self.calcBodyPositions()
            self.headCoordX_list.append(self.headCoord[0,0])
            self.headCoordY_list.append(self.headCoord[0,1])
            """Calculate psi angle and area covered in one trajectory"""
            self.psiAngles_list.append(self.calcPsiAngle())
            """Food routine"""
            foodFoundCoord = self.foodroutine()
            self.foodFound_coordList.append(foodFoundCoord)
        self.count +=1
    
    def calcStepSize(self):
        self.stepSize_arr = np.zeros([len(self.startListX)])
        for i in np.arange(1,len(self.startListX)):
            stepSize = np.sqrt((self.startListX[i]-self.startListX[i-1])**2 + (self.startListY[i]-self.startListY[i-1])**2)
            self.stepSize_arr[i]=stepSize
        return self.stepSize_arr
    
    def calculateTransVsRot(self):
        threshold_transVel = 0.5 #mm/s
        threshold_rotVel = 1 #rad/s        
        self.types_of_movements = []
        for i in range(len(self.all_performedPMs)):
            thrustVel = abs(self.VELO_arr[self.all_performedPMs[i]][0])
            slipVel = abs(self.VELO_arr[self.all_performedPMs[i]][1])
            rotVel = abs(self.VELO_arr[self.all_performedPMs[i]][2])
            if thrustVel + slipVel <= threshold_transVel and rotVel <= threshold_rotVel:
                self.types_of_movements.append('stationary')
            elif thrustVel + slipVel > threshold_transVel and rotVel <= threshold_rotVel:
                self.types_of_movements.append('translational')
            elif rotVel > threshold_rotVel:
                self.types_of_movements.append('rotational')        
        self.movementTypes_proportion = [(self.types_of_movements.count('stationary')/len(self.types_of_movements))*100, 
                                         (self.types_of_movements.count('translational')/len(self.types_of_movements))*100, 
                                         (self.types_of_movements.count('rotational')/len(self.types_of_movements))*100]
        
    def calculateAverageVelocities(self):
        self.allVelocities = self.VELO_arr[np.array(self.all_performedPMs)]
        self.averageVelocities = np.mean(self.allVelocities, axis=0)        
            
    def clearObject(self,startPoint =np.array([[0,0]])):
        self.startPoint    = startPoint
        self.coordinatesX  = []
        self.coordinatesX.append(self.startPoint[0,0])
        self.coordinatesY  = []
        self.coordinatesY.append(self.startPoint[0,1])
        self.allStepsX     = []
        self.allStepsX.append(self.startPoint[0,0])
        self.allStepsY     = []
        self.allStepsY.append(self.startPoint[0,1])
        self.uncorrectedY  = []
        self.uncorrectedY.append(self.startPoint[0,1])
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
        self.headCoordX_list = []
        self.headCoordX_list.append((self.bodyLengthVec/2)[0,0])
        self.headCoordY_list = []
        self.headCoordY_list.append((self.bodyLengthVec/2)[0,1])
        self.tailCoordX_list = []
        self.tailCoordX_list.append((self.bodyLengthVec/2*-1)[0,0])
        self.tailCoordY_list = []
        self.tailCoordY_list.append((self.bodyLengthVec/2*-1)[0,1])
        self.midCoords_list = []
        self.psiAngles_list = []
        
    def simulate(self):
        self.clearObject(startPoint =np.array([[0,0]]))
        self.scatterFood()
        while self.count <= self.maxSteps:
            self.sim_levy()
        self.calcStepSize()
        self.foodFound_total = self.totalFood - self.totalFood_updated
        self.calculateAverageVelocities()        
    
    def simulateMultiple(self):
        self.positionCoordsX_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.positionCoordsY_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.stepSize_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.foodFound_total_allTrials = np.ones((self.nrTrial, 1))*np.nan
        self.thrustVelocities_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.slipVelocities_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.yawVelocities_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.yawAngles_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan  
        self.psiAngles_allTrials = np.ones((self.nrTrial, self.maxSteps+1))*np.nan
        self.performedPMs_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.tailCoordsX_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.tailCoordsY_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.headCoordsX_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        self.headCoordsY_allTrials = np.ones((self.nrTrial, self.maxSteps+2))*np.nan
        c = 0
        while c < self.nrTrial:
            self.simulate()
            c +=1
            print (c)
            self.positionCoordsX_allTrials[c-1,:] = self.startListX[:]
            self.positionCoordsY_allTrials[c-1,:] = self.startListY[:]
            self.stepSize_allTrials[c-1,:] = self.stepSize_arr
            self.foodFound_total_allTrials[c-1,0] = self.foodFound_total
            self.thrustVelocities_allTrials[c-1,:] = self.allVelocities[:,0]
            self.slipVelocities_allTrials[c-1,:] = self.allVelocities[:,1]
            self.yawVelocities_allTrials[c-1,:] = self.allVelocities[:,2]
            self.yawAngles_allTrials[c-1,:] = self.bodyAngles_list
            self.psiAngles_allTrials[c-1,:] = self.psiAngles_list
            self.performedPMs_allTrials[c-1,:] = self.all_performedPMs
            self.tailCoordsX_allTrials[c-1,:] = self.tailCoordX_list[:]
            self.tailCoordsY_allTrials[c-1,:] = self.tailCoordY_list[:]
            self.headCoordsX_allTrials[c-1,:] = self.headCoordX_list[:]
            self.headCoordsY_allTrials[c-1,:] = self.headCoordY_list[:]
            