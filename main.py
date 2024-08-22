# a simple genetic algorithm of ants
# each ant has a little brain that we can evolve.

# start with a 6 digit hex code
# each value will evaluate to a force value
# the value can be from an input to an output
# or an input to an intermediary neuron that can hold a value
# or it can be from that neuron to an output


#each ant brain is randomly created from three values
# each brain will have an in, force and out
# each brain will be connected to inputs like, direction, food in the front direction, collision in the front direction, last direction, etc
# each brain will have outputs like, move forward, turn left, turn right, etc

# we can draw the window with pygame


import random
import math
import time

import pygame

import json
import os
import pygame.display
import numpy as np

import sys

import argparse


class Ant:
    def __init__(self):
        self.brain = []

        self.neurons = [ 0 for i in range(6) ]
        self.fitness = 0
        self.direction = 0
        self.pDirection = 0
        self.x = 0
        self.y = 0
        self.energy = 100
        self.life = 300
        self.FoodConsumed = 0
        self.blockedFront = False
        self.ClossestFood = [-1,-1]

        self.InputSources = [ self.getDirection,self.getPdirection, self.getBlockedFront, self.foodDir, self.foodFront, self.oscillate, self.randomNum, self.GetPherFront, self.closeToFood ]
        self.OutputDestinations = [ self.move, self.turn, self.dropPherimone ]

        self.DropPherAmount = 0
        self.pherFront = 0
        
        self.posHistory = []
        
        self.DebugBrain = False

    def create_brain(self, size=6):
        """Create_Brain : create a random brain"""
        for i in range(size):
           src = random.random() > 0.5
           selectorSrc = random.randint(0, max(len(self.InputSources), len(self.neurons)))
           selectorDst = random.randint(0, max(len(self.OutputDestinations), len(self.neurons)))
           frc = random.random() * 2 - 1
           dest = random.random() > 0.5
           self.brain.append((src, selectorSrc, selectorDst, frc, dest))
           
    def closeToFood(self):
        """closeToFood() : returns true if the ant is close to food"""
        #if closest food is found, return true
        
        isClose = 0
        
        if self.ClossestFood != [-1,-1]:
            isClose = 1
        else:
            isClose = 0
        if self.DebugBrain:
            print(f'isClose: {isClose}')
        return isClose

    def RunBrain(self):
        """RunBrain() : run the brain of the ant"""
        for i in range(len(self.brain)):
            synapseN = self.brain[i]
            if self.DebugBrain:
                print(f'Synapse: {synapseN}')
            src = synapseN[0]  # true or false if the source is an input
            selectorSrc = synapseN[1]  # the selector value
            selectorDst = synapseN[2]  # the selector value
            frc = synapseN[3]  # the force value
            dest = 0
            dest = synapseN[4]  # true or false if the destination is an output
     
            if self.DebugBrain:
                print(f'Brain: {i}  src: {"Input" if src else "Neuron Input"}  selectorSrc: {selectorSrc}  selectorDst: {selectorDst}  frc: {frc}  dest: {"Output" if dest else "Neuron Output"}')
            
            if src:
                # get the value from the input source
                idx = selectorSrc % len(self.InputSources)
                val = self.InputSources[idx]()
                if self.DebugBrain:
                    print(f'Input: {self.InputSources[idx].__doc__} value: {val}')
            else:
                idx = selectorSrc % len(self.neurons)
                val = self.neurons[idx]
                #cosine the value
                val = math.cos(val)
                if self.DebugBrain:
                    print(f'Neuron Input: {idx}  value: {val}')

            
            if self.DebugBrain:
                print(f'val*frc: {val * frc}')


            if dest: # if the destination is an output
                # set the value to the output destination
                outIdx = selectorDst % len(self.OutputDestinations)
                self.OutputDestinations[outIdx](val * frc)
                if self.DebugBrain:
                    print(f'Output: {self.OutputDestinations[outIdx].__doc__} value: {val * frc}')
            else: # if the destination is a neuron
                # set the value to the neuron
                outIdx = selectorDst % len(self.neurons)
                self.neurons[outIdx] += (val * frc)
                if self.DebugBrain:
                    print(f'Neuron Output Set to: {outIdx}  value: {self.neurons[outIdx]}')
            if self.DebugBrain:
                print('____')
        if self.DebugBrain:
            print('-------------------------')


    def randomNum(self):
        """randomNum() : returns a random number between -1 and 1"""
        return random.random() * 2 - 1
    

    def oscillate(self):
        """oscillate() : returns a value between -1 and 1"""
        return math.sin(self.life/10)
    
    def getDirection(self):
        """getDirection() : returns the direction of the ant in -1 to 1 """
        outDir = self.direction % (2*math.pi) / (2*math.pi)
        return outDir

    def getPdirection(self):
        """getPdirection() : returns the previous direction of the ant in -1 to 1 """
        return self.pDirection / (2*math.pi)
    
    def getEnergy(self):
        """getEnergy() : returns the energy of the ant """
        return self.energy
    
    def getLife(self):
        """getLife() : returns the life of the ant """
        return self.life

    def getBlockedFront(self):
        """getBlockedFront() : returns true if the ant is blocked in front of it """
        # figure out the direction of the ant and the tile in front of it
        # if the tile in front of it is a wall return true
        return 1 if self.blockedFront else 0
     
    def GetPherFront(self):
        """GetPherFront() : returns the amount of pheromone in front of the ant """
        # get the amount of pheromone in front of the ant
        # return a value between 0 and 1
        return self.pherFront
    
    def foodDir(self):
        """foodDir() : returns the direction of the closest food """
        # return the relative direction of the closest food in a value between -1 and 1
        # if there is no food return 0
        if self.ClossestFood == [-1,-1]:
            return 0
        
        #if distance to food is far, return 0
        
        #get the direction of the food
        dx = self.ClossestFood[0] - self.x
        dy = self.ClossestFood[1] - self.y
        
        # dist = math.sqrt(dx**2 + dy**2)
        # if dist > 5:
        #     return 0
        
        angle = math.atan2(dy, dx)
        angle = angle - self.direction
        return angle / math.pi
 
    def foodFront(self):
        """foodFront() : returns the distance to the closest food in front of the ant """
        #get distance to closest food, then map this distance to a value between 0 and 1
        # limit distance to 5 tiles from the ants position
        if self.ClossestFood == [-1,-1]:
            return 0
        dx = self.ClossestFood[0] - self.x
        dy = self.ClossestFood[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 5:
            return 0
        viewAngle = 10 #degrees of view angle
        #check the current ant angle vs the angle to the food
        # if the difference is less than the view angle then the food is in front of the ant
        angleToFood = math.atan2(dy, dx)
        angleToFood = abs(angleToFood - self.direction)
        if angleToFood < viewAngle:
            return 1 - dist / 5
        
        return 0
    
    def move(self,ammount):
        """move() : move the ant in the direction it is facing """
        #limit ammount to 5
        if ammount > .8:
            ammount = .8
        if ammount < -.8:
            ammount = -.8
        if self.blockedFront:
            ammount = 0
            self.life -= 1# penalize the ant for hitting a wall
        self.x += math.cos(self.direction) * ammount
        self.y += math.sin(self.direction) * ammount
        
        
        
        self.energy -= 1
    
    def turn(self, direction):
        """turn() : turn the ant in a direction """
        self.direction += direction
        self.energy -= 1

    def dropPherimone(self,ammt):
        """dropPherimone() : drop pherimone at the current location """
        #limit to 1 or -1
        if ammt > 1:
            ammt = 1
        if ammt < 0:
            ammt = 0
            
        self.DropPherAmount += ammt
        
        #drop pherimone at the current location
        
        return

class WorldGrid:
    def __init__(self, width, height):
        """ a grid to hold stuff like food, walls, etc """
        self.grid = []
        for i in range(width):
            row = []
            for j in range(height):
                row.append([])
            self.grid.append(row)
        return
    
    def RemoveVal(self, x, y):
        """ remove a value from the grid """
        if x < 0 or x >= len(self.grid):
            return False
        if y < 0 or y >= len(self.grid[x]):
            return False
        #if itsalready empty, return false
        if self.grid[x][y] == []:
            return False
        self.grid[x][y] = []
        return True
    
    def SetVal(self, x, y, val):
        if x < 0 or x >= len(self.grid):
            return
        if y < 0 or y >= len(self.grid[x]):
            return
        self.grid[x][y] = val
        return

    def GetVal(self, x, y):
        if x < 0 or x >= len(self.grid):
            return None
        if y < 0 or y >= len(self.grid[x]):
            return None
        return self.grid[x][y]
    
    def Clear(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.grid[i][j] = []
        return
    
    def listActive(self):
        active = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != []:
                    active.append([i, j, self.grid[i][j]])
        return active

class AntColony:
    def __init__(self, _screenSize, _maxAnts, _tileSize):
        self.maxAnts = _maxAnts
        self.ants = []
        #hive pos is 80percent in the corner right
        self.screenSize = _screenSize
        self.FollowNextAnt = False
        #40 pixels per tile
        self.TileSize = _tileSize
        GridSize = [int(self.screenSize[0]/self.TileSize), int(self.screenSize[1]/self.TileSize)]
        self.hivePos = [int(GridSize[0]*0.5), int(GridSize[1]*0.5)]
        self.foodGrid = WorldGrid(GridSize[0], GridSize[1])
        self.wallGrid = WorldGrid(GridSize[0], GridSize[1])
        self.pheromoneGrid = WorldGrid(GridSize[0], GridSize[1])
        self.width = GridSize[0]
        self.height = GridSize[1]

        self.create_world()
        self.StartTime = time.time()
        
        self.BestAnts = []
        self.LastBestAnts = []
        
        self.TopPheromone = 0

        self.LastSave = time.time()
        
        self.UpdateTime = 0
        
        self.totalDeadAnts = 0
        self.topFoodFound = 0
        self.totalSteps = 0
        
                 
    def WorldToScreen(self, apos):
        ax = apos[0]/self.width * self.screenSize[0]
        ay = apos[1]/self.height * self.screenSize[1]
        return (ax, ay)

    def add_ant(self, brain=None, startP=None):
        ant = Ant()
        if self.FollowNextAnt:
            ant.DebugBrain = True
            self.FollowNextAnt = False
        
        ant.x = random.randint(0, self.width)
        ant.y = random.randint(0, self.height)
        if startP:
            ant.x = startP[0]
            ant.y = startP[1]
        
        if brain:
            ant.brain = brain
        else:
            ant.create_brain(random.randint(6, 24))
        self.ants.append(ant)
        # print(f'added new ant with brain {ant.brain}')

    def add_food(self, pos=None):
        """add to the food grid in a random spot"""
        foodX = random.randint(0, self.width)
        foodY = random.randint(0, self.height)
        if pos:
            foodX = pos[0]
            foodY = pos[1]

        self.foodGrid.SetVal(foodX, foodY, 1)
 
    def add_wall(self, pos=None):
        """add to the wall grid in a random spot"""
        wallX = random.randint(0, self.width)
        wallY = random.randint(0, self.height)
        if pos:
            wallX = pos[0]
            wallY = pos[1]
        self.wallGrid.SetVal(wallX, wallY, 1)

    def create_world(self):
        for i in range(self.maxAnts):
            self.add_ant(brain=None, startP=self.hivePos)
        for i in range(10):
            self.add_food()
        # make a cross of walls 70 percent of width and height
        # startX = int(self.width * 0.15)
        # endX = int(self.width * 0.85)
        # startY = int(self.height * 0.15)
        # endY = int(self.height * 0.85)
        
        # for i in range(startX, endX):
        #     self.wallGrid.SetVal(i, int(self.height/2), 1)
        # for i in range(startY, endY):
        #     self.wallGrid.SetVal(int(self.width/2), i, 1)
            
        
        #just make some random walls around
        #width*height * .3
        numWalls = int(self.width * self.height * .08)
        
            
        #add some random walls
        for i in range(numWalls):
            self.add_wall()
           
    def Repopulate(self):
        """repopulate the ants with the best ants"""
        currentAnts = len(self.ants)
        bestAntNum = len(self.BestAnts)
        
        if bestAntNum == 0:
            #just make new ants
            while len(self.ants) < self.maxAnts:
                self.add_ant(brain=None, startP=self.hivePos)
            return
        #sort and trim the best ants to top 50 ants
        self.BestAnts = sorted(self.BestAnts, key=lambda x: x["food"], reverse=True)
        self.BestAnts = self.BestAnts[:50]
        probBest = .2
        
        #create new ants from the best ants
        while len(self.ants) < self.maxAnts:
            # ant = Ant()
            bestAntScore = self.BestAnts[0]["food"]
            
            #make more new ants if the best ant score is less than 10
            
            # if bestAntScore < 30:
            #     probBest = 1-(bestAntScore / 60) # the higher the score, the more likely we add a best ant
            if bestAntScore < 2:
                probBest = .9
            elif bestAntScore < 20:
                probBest = .5
            else:
                probBest = .2
            
            if random.random() < probBest: # if less then probBest, add a random ant
                self.add_ant(brain=None, startP=self.hivePos)
            else: # add a best ant or pick two best ants as parents
                if random.random() > 0.5: #fifty fifty chance to mutate the brain or parent
                    newBrain = self.BestAnts[random.randint(0, len(self.BestAnts)-1)]["brain"].copy()
                    if random.random() > 0.5: #mutate the brain or not
                        newBrain = self.MutateBrain(newBrain)
                        
                    self.add_ant(brain=newBrain, startP=self.hivePos)

                else:
                    #join two parents together
                    par1 = self.BestAnts[random.randint(0, len(self.BestAnts)-1)]["brain"].copy()
                    par2 = self.BestAnts[random.randint(0, len(self.BestAnts)-1)]["brain"].copy()
                    #select random parts of the brain from each parent
                    newBrain = []
                    for i in range(min(len(par1), len(par2))): # cannot be longer than the shortest brain
                        if random.random() > 0.5:
                            newBrain.append(par1[i])
                        else:
                            newBrain.append(par2[i])
                    self.add_ant(brain=newBrain, startP=self.hivePos)
                    
        return {'ants':len(self.ants), 'probbest':probBest}            
    def ReplenishFood(self):
        """add food to the grid"""
        
        # specific quadrant for each food every minute
        timeSinceStart = time.time() - self.StartTime
        #every 5 minutes switch the food to a new quadrant
        quadrant = int(timeSinceStart / 240) % 4 # every 5 minutes
        quads = []
        for i in range(2):
            for j in range(2):
                quad = [[i*self.width/2, j*self.height/2], [i*self.width/2, (j+1)*self.height/2], [(i+1)*self.width/2, j*self.height/2], [(i+1)*self.width/2, (j+1)*self.height/2]]
                quads.append(quad)
        # count current food and add more as needed
        currentFood = len(self.foodGrid.listActive())
        while currentFood < 50: # food scaricity produces more competition for food
            #find a random spot in the quadrant
            qLeft = int(min(quads[quadrant][0][0], quads[quadrant][3][0]))
            qRight = int(max(quads[quadrant][0][0], quads[quadrant][3][0]))
            # print(f'qLeft: {qLeft}, qRight: {qRight}')
            x = random.randint(qLeft, qRight)
            qBottom = int(min(quads[quadrant][1][1], quads[quadrant][2][1]))
            qTop = int(max(quads[quadrant][1][1], quads[quadrant][2][1]))
            # print(f'qBottom: {qBottom}, qTop: {qTop}')
            y = random.randint(qBottom, qTop)
            foodPos = [x, y]
            # print(f'adding food at {foodPos}')
            #create littel clusters of 20 food
            for i in range(20):
                foodPosRand = [foodPos[0] + random.randint(-5, 5), foodPos[1] + random.randint(-5, 5)]
                #dont drop within 20 tiles of the hive
                distToHive = math.sqrt((foodPosRand[0] - self.hivePos[0])**2 + (foodPosRand[1] - self.hivePos[1])**2)
                if distToHive > 20:
                    #dont drop on a wall
                    if self.wallGrid.GetVal(foodPosRand[0], foodPosRand[1]) == []:
                        self.add_food(foodPosRand)
                        currentFood = len(self.foodGrid.listActive())
            
    def MutateBrain(self, brain):
        """mutate the brain by changing one of the values"""
        #select a random value to change
        #change the src, selector, frc or dest
        numChanges = random.randint(1, 3)
        for i in range(numChanges):
            idx = random.randint(0, len(brain)-1)
            src = brain[idx][0]
            selectorSrc = brain[idx][1]
            selectorDst = brain[idx][2]
            frc = brain[idx][3]
            dest = brain[idx][4]
            
            #change one of the values
            change = random.randint(0, 3)
            if change == 0:
                src = not src
            elif change == 1:
                selector = random.randint(0, 1000)
            elif change == 2:
                frc = random.random() * 2 - 1
            elif change == 3:
                dest = not dest
                
            brain[idx] = (src, selectorSrc, selectorDst, frc, dest)
        return brain
            
    def update(self):
        self.totalSteps += 1
           
        startTime = time.time()
        for ant in self.ants:
            antPos = (int(ant.x), int(ant.y))
            #normal vector of the direction of the ant
            
            antDir = ant.direction % (2*math.pi)
            #check for NAN values
            if math.isnan(antDir):
                #reset the ant direction
                ant.direction = 0
            normVec = (math.cos(antDir), math.sin(antDir))
            # print(f'antPos: {antPos}, normVec: {normVec}')
            frontPos = (int(antPos[0] + normVec[0]), int(antPos[1] + normVec[1]))
            # print(f'frontPos: {frontPos}')
            #check for NAN values
            if math.isnan(frontPos[0]) or math.isnan(frontPos[1]):
                #reset the ant to the hive
                ant.x = self.hivePos[0]
                ant.y = self.hivePos[1]
                ant.direction = 0
            if self.wallGrid.GetVal(frontPos[0], frontPos[1]) == 1:
                ant.blockedFront = True
            else:
                ant.blockedFront = False
        
            if frontPos[0] < 0 or frontPos[0] >= self.width:
                ant.blockedFront = True
            if frontPos[1] < 0 or frontPos[1] >= self.height:
                ant.blockedFront = True
            
            # find the closest food
            closest = 1000000
            closestFood = [-1,-1]
            for food in self.foodGrid.listActive():
                dist = math.sqrt((ant.x - food[0])**2 + (ant.y - food[1])**2)
                if dist < closest:
                    closest = dist
                    if dist < 20:
                        closestFood = food
            if closestFood != [-1,-1]:
                ant.ClossestFood = closestFood
                
            ant.RunBrain()
            ant.pDirection = float(ant.direction)
            #if moved greater than 1 tile, add to the history
            lastknownPos = ant.posHistory[-1] if len(ant.posHistory) > 0 else [ant.x, ant.y]
            moveDist = math.sqrt((ant.x - lastknownPos[0])**2 + (ant.y - lastknownPos[1])**2)
            if abs(moveDist) > 1:
                ant.posHistory.append([ant.x, ant.y])

            ant.life -= 1
            if ant.life <= 0: #DEAD ANT
                self.ants.remove(ant)
                #check how much food the ant consumed
                foodConsumed = ant.FoodConsumed
                antBrain = ant.brain
                self.totalDeadAnts += 1
                if foodConsumed > 0:
                    self.BestAnts.append({"food":foodConsumed, "brain":antBrain})
            # topFoodCount = self.BestAnts[0]["food"] if len(self.BestAnts) > 0 else 0
            
            #check if the ant has a closestFood value, if so detect the distance and if close enough, consume the food
            
            antClosestFood = ant.ClossestFood
            if antClosestFood != [-1,-1]:
                distToFood = math.sqrt((ant.x - antClosestFood[0])**2 + (ant.y - antClosestFood[1])**2)
                if distToFood < 2: #give a little leeway
                    if self.foodGrid.RemoveVal(antClosestFood[0], antClosestFood[1]):
                        # print(f'removing food at {antClosestFood}')
                        ant.ClossestFood = [-1,-1]
                        ant.energy += 10
                        ant.FoodConsumed += 1
                        ant.life += 10 #reward the ant for finding food
                        # print(f'ant consumed food, food consumed: {ant.FoodConsumed}')
                        if ant.FoodConsumed > self.topFoodFound:
                            print(f'New top ant!!: {ant.FoodConsumed}')
                            self.topFoodFound = ant.FoodConsumed
                        
            # if self.foodGrid.GetVal(int(ant.x), int(ant.y)) == 1: #ANT ON FOOD
            #     ant.energy += 10
            #     ant.FoodConsumed += 1
            #     ant.life += 30 #reward the ant for finding food
            #     # print(f'ant consumed food, food consumed: {ant.FoodConsumed}')
            #     if ant.FoodConsumed > topFoodCount:
            #         print(f'New top ant!!: {ant.FoodConsumed}')
            #     # self.foodGrid.SetVal(int(ant.x), int(ant.y), 0)
            #     self.foodGrid.RemoveVal(int(ant.x), int(ant.y))
            # #drop pheromone if the ant has any
            if ant.DropPherAmount > 0:
                getCurrentPher = self.pheromoneGrid.GetVal(int(ant.x), int(ant.y))
                if getCurrentPher == [] or getCurrentPher == None:
                    getCurrentPher = 0

                newAmmt = getCurrentPher + ant.DropPherAmount
                if newAmmt < 10:
                    self.pheromoneGrid.SetVal(int(ant.x), int(ant.y), newAmmt) 
                ant.DropPherAmount = 0
                
            #check if front pos is on a pheromone
            pherVal = self.pheromoneGrid.GetVal(int(frontPos[0]), int(frontPos[1]))
            
            if pherVal != [] and pherVal != None:
                if pherVal > 0:
                    # print(f'pherVal: {pherVal}')
                    ant.pherFront = pherVal
                

                
            
            # for food in self.food:
            #     if ant.x == food[0] and ant.y == food[1]:
            #         ant.energy += 10
            #         self.food.remove(food)
            # for wall in self.walls:
            #     if ant.x == wall[0] and ant.y == wall[1]:
            #         ant.energy -= 10

        #remove old pherimone cells, anything less than 0
        for pher in self.pheromoneGrid.listActive():
            if pher[2] <= 0:
                self.pheromoneGrid.RemoveVal(pher[0], pher[1])
            else:
                #decay the pheromone
                self.pheromoneGrid.SetVal(pher[0], pher[1], pher[2]-.005)

        self.ReplenishFood()
        repop_result = self.Repopulate()
        endTime = time.time()
        self.UpdateTime = endTime - startTime
        
        if self.totalSteps % 1000 == 0:
            print("------------------report-----------------")
            print(f'Steps So Far: {self.totalSteps}')
            print(f'Total Ants: {len(self.ants)}')
            print(f'Total Dead Ants: {self.totalDeadAnts}')
            print(f'Top Food Found: {self.topFoodFound}')
            # print(f'Top Pheromone: {self.TopPheromone}')
            print(f'Update Time: {self.UpdateTime}')
            print(f'Repopulate Probiability of best ant: {repop_result["probbest"]}')
            print("----------------------------------------")


    def LoadBestAnts(self):
        """load the best ants from a file"""
        bestAntsFound = []
        #find and load every file you can find. add all ants and select the top 50
        
        searchFolder = 'dataSave'
        for root, dirs, files in os.walk(searchFolder):
            for file in files:
                if file.endswith('.json'):
                    with open(f'{searchFolder}/{file}', 'r') as f:
                        data = f.read()
                        data = json.loads(data)
                        bestAnts = data["BestAnts"]
                        bestAntsFound.extend(bestAnts)
        #sort the best ants
        bestAntsFound = sorted(bestAntsFound, key=lambda x: x["food"], reverse=True)
        #randomize best ants
        numNewAnts = 20
        bestAntsFound = bestAntsFound[:numNewAnts]
        
        random.shuffle(bestAntsFound)
        #add these ants to the game
        
        #top ants getmore new ants
       
        for i in range(int(numNewAnts)):
            antBrain = bestAntsFound[0]["brain"].copy()
            if len(antBrain[0])==4: ## old version onlyhad 3 values
                self.add_ant(brain=antBrain, startP=self.hivePos)
                
        print(f'Loaded {numNewAnts} best ants from file')
        return

    def drawAnts(self,screen, isPi=False):
        
        #preload the best ants from a file
        #find all files

        
        #update 10 times
        #for i in range(2):
        # self.update()

        maxPher = 0
        for pher in self.pheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal > maxPher:
                maxPher = pherVal
            
        self.TopPheromone = maxPher
        for pher in self.pheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal <= 0:
                continue
            ppxy = self.WorldToScreen([pher[0], pher[1]])
            #map color to the pheromone value between 0 and 1000
            #limit pherVal to 0

            colorVal = pherVal / maxPher * 255.0
            #range between 10 and 255
            if colorVal < 10:
                colorVal = 10
            
            # if pherVal > 0:
            #     print(f'pherVal: {pherVal}  ColorVal: {colorVal}')
            #brightness of the pheromone is alpha
            ppxy = (int(ppxy[0]), int(ppxy[1]))
            pygame.draw.rect(screen, (0,0,colorVal), ((ppxy[0]), (ppxy[1]), int(self.TileSize), int(self.TileSize)))
        timeDelta = time.time() - self.LastSave
        timeDistance = 60 #save every minute
        if isPi:
            timeDistance = 60*60 # hour #save every hour for the pi

        if timeDelta > timeDistance:
    
            #save the best ants to a file
            self.saveData()
            #also save an image of the screen\
            tCode = time.strftime("%Y%m%d-%H%M%S")
            if isPi == False:
                pygame.image.save(screen, f'dataSave/{tCode}.png')
            self.LastSave = time.time()

        for food in self.foodGrid.listActive():
            if food[2] == 0:
                continue
            fpxy = self.WorldToScreen(food)
            fpxy = (int(fpxy[0]), int(fpxy[1]))
            # pygame.draw.rect(screen, (0, 200, 0), (fpxy[0], fpxy[1], self.TileSize, self.TileSize))
            #small green circle empty wih no fill green border
            cx = fpxy[0] + self.TileSize/2
            cy = fpxy[1] + self.TileSize/2
            pygame.draw.circle(screen, (0, 200, 0), (int(cx), int(cy)), int(self.TileSize/2), width=1)
            
        for wall in self.wallGrid.listActive():
            wpxy = self.WorldToScreen(wall)
            wpxy = (int(wpxy[0]), int(wpxy[1]))
            # pygame.draw.rect(screen, (100,100,100), (wpxy[0], wpxy[1], self.TileSize, self.TileSize))
            # small gray hollow box with x inside
            pygame.draw.rect(screen, (100,100,100), (wpxy[0], wpxy[1], self.TileSize, self.TileSize), width=1)
            # pygame.draw.line(screen, (100,100,100), (wpxy[0], wpxy[1]), (wpxy[0]+self.TileSize-1, wpxy[1]+self.TileSize-1))
            # pygame.draw.line(screen, (100,100,100), (wpxy[0]+self.TileSize, wpxy[1]), (wpxy[0]-1, wpxy[1]+self.TileSize-1))
            

        for ant in self.ants:
            pxy = self.WorldToScreen((ant.x, ant.y))
            pxy = (int(pxy[0]), int(pxy[1]))
            #square if the ant is blocked
            
            #antColor is based on the first three force values of the ant brain
            RV = abs(ant.brain[0][2]) * 255
            GV = abs(ant.brain[1][2]) * 255
            BV = abs(ant.brain[2][2]) * 255
            
            #limit RGB values to 0-255
            if RV > 255:
                RV = 255
            if GV > 255:
                GV = 255
            if BV > 255:
                BV = 255

            if RV < 0:
                RV = 0
            if GV < 0:
                GV = 0
            if BV < 0:
                BV = 0
                            
            #if ant.blockedFront:
            pygame.draw.rect(screen, (RV,GV,BV), (pxy[0]-2.5, pxy[1]-2.5, 5, 5))
            #pos
            # print(f'ant pos: {pxy}')
            # else:
                # pygame.draw.circle(screen, (RV,GV,BV), (int(pxy[0]), int(pxy[1])), 2.5)
                
            #draw the path of the ant
            # for i in range(len(ant.posHistory)-1):
            #     p1 = self.WorldToScreen(ant.posHistory[i])
            #     p2 = self.WorldToScreen(ant.posHistory[i+1])
            #     pygame.draw.line(screen, (255, 0, 0), (p1[0], p1[1]), (p2[0], p2[1]))
            #draw the direction of the ant with a line
            # pygame.draw.line(screen, (0, 0, 0), (pxy[0], pxy[1]), (pxy[0] + math.cos(ant.direction) * 10, pxy[1] + math.sin(ant.direction) * 10))

            #debug show a line to the closest food
            # if ant.ClossestFood != [-1,-1]:
            #     fpxy = self.WorldToScreen(ant.ClossestFood)
            #     pygame.draw.line(screen, (0, 255, 0), (pxy[0], pxy[1]), (fpxy[0], fpxy[1]))
            if ant.DebugBrain:
                #display the brain in a box in the right bottom corner
                #make three columns, one for all the optional inputs, one for the neurons and one for the outputs
                
                #draw rect box, gray bg
                pygame.draw.rect(screen, (50,50,50), (self.screenSize[0]-300, self.screenSize[1]-300, 300, 300))
                #draw circles for the inputs
                column_1 = 50
                column_2 = 150
                column_3 = 250
                
                for i in range(len(ant.InputSources)):
                    pygame.draw.circle(screen, (255, 255, 255), (self.screenSize[0]-250, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the input in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.InputSources[i]()}', True, (255,255,255))
                    screen.blit(text, (self.screenSize[0]-250, self.screenSize[1]-250 + i*20))
                    
                    
                for i in range(len(ant.neurons)):
                    pygame.draw.circle(screen, (255, 255, 255), (self.screenSize[0]-150, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the neuron in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.neurons[i]}', True, (255,255,255))
                    screen.blit(text, (self.screenSize[0]-150, self.screenSize[1]-250 + i*20))
                    
                
                for i in range(len(ant.OutputDestinations)):
                    pygame.draw.circle(screen, (255, 255, 255), (self.screenSize[0]-50, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the output in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.OutputDestinations[i]}', True, (255,255,255))
                    screen.blit(text, (self.screenSize[0]-50, self.screenSize[1]-250 + i*20))
                    
                #draw the lines between the circles
                for BR in ant.brain:
                    src = BR[0]
                    srcSel = BR[1]
                    dstSel = BR[2]
                    frc = BR[3]
                    dest = BR[4]
                    
                    
                    
                    
                    yPosStart = 0
                    yPosEnd = 0
                    finalForce = frc
                    
                    #figure out what column the src and dest are in
                    if src: # tru if the source is an input not a neuron so column and not column_2 for the input
                        xStart = column_1
                        #figure out the y position because we know the number of inputs
                        yPosStart = srcSel % len(ant.InputSources) * 20
                    else:
                        xStart = column_2
                        yPosStart = srcSel % len(ant.neurons) * 20
                        finalForce = frc * ant.neurons[srcSel % len(ant.neurons)]
                    #next figure out if the dest is an output or a neuron, neurons are column_2 and outputs are column_3
                    if dest:
                        xEnd = column_3
                        yPosEnd = dstSel % len(ant.OutputDestinations) * 20
                    else:
                        xEnd = column_2
                        yPosEnd = dstSel % len(ant.neurons) * 20
                        
                    #green line if frc is positive
                    lineColor = (0, 255, 0)
                    #red line if frc is negative
                    if finalForce < 0:
                        lineColor = (255, 0, 0)
                        
                    #line thickness is the force
                    lineThickness = abs(finalForce) * 5
                    
                    if lineThickness > 5:
                        lineThickness = 5
                        
                    
                    
                    #draw the line from the src to the neuron or output
                    # pygame.draw.line(screen, lineColor, (xStart, yPos), (xEnd, yPos))
                    pygame.draw.line(screen, lineColor, (xStart, self.screenSize[1]-250 + yPosStart), (xEnd, self.screenSize[1]-250 + yPosEnd), int(lineThickness))
                    
                    # #draw the line from the src to the neuron or output
                    # if src:
                    #     selIdx = srcSel % len(ant.InputSources)
                    #     destIdx = dstSel % len(ant.neurons)
                    #     if dest:
                    #         destIdx = dstSel % len(ant.OutputDestinations)
                    #         pygame.draw.line(screen, lineColor, (self.screenSize[0]-250, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-150, self.screenSize[1]-250 + destIdx*20))
                    #     else:
                    #         pygame.draw.line(screen, lineColor, (self.screenSize[0]-250, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-150, self.screenSize[1]-250 + destIdx*20));
                    #     # pygame.draw.line(screen, lineColor, (self.screenSize[0]-250, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-150, self.screenSize[1]-250 + destIdx*20))
                    # else:
                    #     selIdx = srcSel % len(ant.neurons)
                    #     destIdx = dstSel % len(ant.neurons)
                    #     if dest:
                    #         destIdx = dstSel % len(ant.OutputDestinations)
                    #         pygame.draw.line(screen, lineColor, (self.screenSize[0]-150, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-50, self.screenSize[1]-250 + destIdx*20))
                    #     else:
                    #         pygame.draw.line(screen, lineColor, (self.screenSize[0]-150, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-150, self.screenSize[1]-250 + destIdx*20))
                    #     # pygame.draw.line(screen, lineColor, (self.screenSize[0]-150, self.screenSize[1]-250 + selIdx*20), (self.screenSize[0]-150, self.screenSize[1]-250 + destIdx*20))
                        
                    
                
                   

        #show some stats on a box in the left top corner
        dataShow = self.BestAnts[:30]
        #show the top 10
        for i, ant in enumerate(dataShow):
            
            #make a simple brain view string like this T:3:5:.14:F when the array is [(True, 3, 5, .1402042451273, False)]
            
            brainStr = ''
            for BR in ant["brain"]:
                src = 'T' if BR[0] else 'F'
                srcSel = BR[1]
                dstSel = BR[2]
                # // round force
                frc = round(BR[3], 1)
                dest = 'T' if BR[4] else 'F'
                brainStr += f'{src}:{srcSel}:{dstSel}:{frc}:{dest} '
            
            textV = f'Ant {i+1}: Food: {ant["food"]}  Brain: {brainStr}'
            #color white
            #clip the text to 50 characters
            textV = textV[:50]
            
            font = pygame.font.Font(None, 20)
            text = font.render(textV, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i*12))
            
        
    def saveData(self):
        
        #check to see if the data chenged from last time
        if self.BestAnts == self.LastBestAnts:
            print('No changes to save')
            return
        
        self.LastBestAnts = self.BestAnts
        
        saveObj = {"BestAnts":self.BestAnts}
        #today Timecode
        tCode = time.strftime("%Y%m%d-%H%M%S")
        pathSave = f'dataSave/{tCode}.json'
        print(f'Saving to file: {tCode}.json')
        try:
            with open(pathSave, 'w') as f:
                f.write(json.dumps(saveObj, indent=4))
        except Exception as e:
            print(f'Error saving file: {e}')
        return

class Game:
    def __init__(self):
        print('Starting Game')
        #get arguments and see if it is a raspberry pi
        self.isPi = False
        
        self.screenSize = (1000, 1000)
        #using argparse
        parser = argparse.ArgumentParser(description='Run the ant simulation')
        parser.add_argument('--pi', action='store_true', help='Run on a Raspberry Pi')
        args = parser.parse_args()
        if args.pi:
            self.isPi = True
            print('Running on Raspberry Pi')
            self.screenSize = (480, 1920)
        
        # if len(sys.argv) > 1:
        #     print('Arguments: ', sys.argv[1])
        #     if sys.argv[1] == "pi":
        #         isPi = True
        #         print('Running on Raspberry Pi')

        pygame.init()
        
                
        if self.isPi:
            print('Starting display on PI')
            pygame.display.init()
            pygame.display.list_modes()
            os.environ["SDL_VIDEODRIVER"] = "rbcon" # or maybe 'fbcon'
            #set display environment variables
            os.environ["DISPLAY"] = ":0.0"
            # Set up display
            self.screen = pygame.display.set_mode(self.screenSize, pygame.FULLSCREEN)
            # Initialize Pygame
            print('Hiding cursor')
            pygame.mouse.set_visible(False) # Hide cursor here
        
        else:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "-1100,0"
            self.screen = pygame.display.set_mode(self.screenSize)
            
            
        self.clock = pygame.time.Clock()
        #open on second screen
        maxAnts = 300
        tileSize = 10
        if self.isPi:
            maxAnts = 40
            tileSize = 25
        self.antColony = AntColony(self.screenSize, maxAnts, tileSize)
        # self.antColony.LoadBestAnts()
        #first run update 20000 times
        lastPercent = 0
        if self.isPi == False:
            newAnts = 1000
            #add new ants before training
            for i in range(newAnts):
                self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            numRuns = 20
            maxFoodFound = 0
            for i in range(numRuns):
            # numRuns = 0
            # while maxFoodFound < 10:
                # numRuns += 1
                #print every 10 percent
                # percentN = int(i/numRuns * 100)
                # if percentN != lastPercent:
                if i % 100 == 0:
                    # print(f'Training: {percentN}%')
                    print(f'Runs: {i}  MaxFood: {maxFoodFound}')
                    print(f"Total Dead Ants: {self.antColony.totalDeadAnts}")
                    # print(f'Time(ms): {self.antColony.UpdateTime * 1000}')
                    # lastPercent = percentN
                
                self.antColony.update()
                if len(self.antColony.BestAnts) > 0:
                    maxFoodFound = self.antColony.BestAnts[0]["food"]
                else:
                    maxFoodFound = 0
    
    def run(self):
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((25,25,25))

            self.antColony.update()
            self.antColony.drawAnts(self.screen, isPi=self.isPi)

            keys = pygame.key.get_pressed()

            if keys[pygame.K_s]:
                self.antColony.saveData()
                
            if keys[pygame.K_u]:
                percentDone = 0
                for i in range(1000):
                    nowPercent = int(i/1000 * 100)
                    if nowPercent != percentDone:
                        print(f'FAST TRAINING: {nowPercent}%')
                        percentDone = nowPercent
                    self.antColony.update()
            
            #if p is pressed, pick the next ant to debug the brain
            if keys[pygame.K_p]:
                # for ant in self.antColony.ants:
                #     ant.DebugBrain = False
                #     break
                self.antColony.FollowNextAnt = True
                
            #add key will add 1000 ants
            if keys[pygame.K_a]:
                for i in range(1000):
                    self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            
            fps = self.clock.get_fps()
            text = f'FPS: {fps}'
            font = pygame.font.Font(None, 26)
            text = font.render(text, True, (255, 255, 255))
            self.screen.blit(text, (self.screenSize[0]-100, 10))
                    
            pygame.display.flip()
            # self.clock.tick(120)
            self.clock.tick()

        pygame.quit()

if __name__ == "__main__":
    print('Starting Simulation')
    game = Game()
    game.run()
