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

# Pi-specific imports
try:
    from sugarpie import pisugar
    PISUGAR_AVAILABLE = True
except ImportError:
    PISUGAR_AVAILABLE = False


def BrainToColor(brain):
    """Convert a brain to a color"""
    # antColor is based on the FORCE values of the brain
# average every 3rd value of the brain for red, green, blue

    RV = 0
    GV = 0
    BV = 0
    NcNT = 0
    for i in range(0, len(brain), 3):
        if i+2 < len(brain):
            NcNT += 1
            RV += abs(brain[i][3])
            GV += abs(brain[i+1][3])
            BV += abs(brain[i+2][3])
        
    RV = RV / (NcNT) 
    GV = GV / (NcNT) 
    BV = BV / (NcNT)
    RV = RV * 255
    GV = GV * 255
    BV = BV * 255
    
    
    #need to add some contrast since the average turnes out to be gray
    contrastAmt = 1.8
    
    # multiply by contrast amount and subtract 128
    RV = RV * contrastAmt - 128
    GV = GV * contrastAmt - 128
    BV = BV * contrastAmt - 128
    
    
    # Limit RGB values to 0-255
    RV = min(max(RV, 0), 255)
    GV = min(max(GV, 0), 255)
    BV = min(max(BV, 0), 255)
    
    
    return [RV, GV, BV]


    
    # return [RV, GV, BV]

class Ant:
    def __init__(self, colony):
        self.colony = colony  # Add this line to store a reference to the colony

        self.brain = []
        self.antID = [] # a unique id for the ant, if it is a mutation, new ant, clone or child. Clones also get a count

        self.neurons = [ 0 for i in range(6) ]
        self.fitness = 0
        self.direction = 0
        self.pDirection = 0
        self.x = 0
        self.y = 0
        self.energy = 100
        self.life = 100
        self.FoodConsumed = 0
        self.blockedFront = False
        self.ClossestFood = [-1,-1]

        self.carryingFood = False
        
        self.FarthestTraveled = 0 # the farthest the ant has traveled from the hive give a fitness bonus when dead


        #self.InputSources = [ self.getDirection,self.getPdirection, self.getBlockedFront, self.foodDir, self.foodFront, self.oscillate, self.randomNum, self.GetPherFront, self.closeToFood ]
        self.InputSources = [
            self.getDirection, self.getPdirection, self.getBlockedFront,
            self.foodDir, self.foodFront, self.oscillate, self.randomNum,
            self.GetPherFront, self.closeToFood,
            self.GetPherLeft, self.GetPherRight, self.GetPherBack,
            self.getHiveDirection, self.getHiveDistance,
            self.isCarryingFood  # Add here
        ]

       
        self.OutputDestinations = [ self.move, self.turn, self.dropPherimone ]

        self.DropPherAmount = 0
        self.pherFront = 0
        
        self.posHistory = []
        
        self.DebugBrain = False
        
        self.Color = []
        


    def create_brain(self, size=12):
        """Create_Brain : create a random brain"""
        brainSize = random.randint(6, size)
        for i in range(brainSize):
            src = random.random() > 0.5
            dest = random.random() > 0.5
            if src:
                selectorSrc = random.randint(0, len(self.InputSources) - 1)
            else:
                selectorSrc = random.randint(0, len(self.neurons) - 1)
            if dest:
                selectorDst = random.randint(0, len(self.OutputDestinations) - 1)
            else:
                selectorDst = random.randint(0, len(self.neurons) - 1)
            frc = random.uniform(-1, 1)
            self.brain.append((src, selectorSrc, selectorDst, frc, dest))
        self.SetColor()
           
    def SetColor(self):
        """SetColor() : set the color of the ant"""
        COLOR = BrainToColor(self.brain)

        self.Color = COLOR
           
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
                if val > 1:
                    val = 1
                elif val < -1:
                    val = -1
                    
                if self.DebugBrain:
                    print(f'Neuron Input: {idx}  value: {val}')
                #cosine the value
                val = math.cos(val)
                # val = np.tanh(val) #tanh the value because it is between -1 and 1


            
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
                    print(f'Neuron Output {outIdx} Set to: value: {self.neurons[outIdx]}')
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
    
    def GetPherLeft(self):
        """GetPherLeft() : returns the amount of pheromone to the left of the ant """
        angle = self.direction - math.pi / 2
        pos = (int(self.x + math.cos(angle)), int(self.y + math.sin(angle)))
        pherVal = self.colony.pheromoneGrid.GetVal(pos[0], pos[1])
        return pherVal if pherVal else 0

    def GetPherRight(self):
        """GetPherRight() : returns the amount of pheromone to the right of the ant """
        angle = self.direction + math.pi / 2
        pos = (int(self.x + math.cos(angle)), int(self.y + math.sin(angle)))
        pherVal = self.colony.pheromoneGrid.GetVal(pos[0], pos[1])
        return pherVal if pherVal else 0

    def GetPherBack(self):
        """GetPherBack() : returns the amount of pheromone behind the ant """
        angle = self.direction + math.pi
        pos = (int(self.x + math.cos(angle)), int(self.y + math.sin(angle)))
        pherVal = self.colony.pheromoneGrid.GetVal(pos[0], pos[1])
        return pherVal if pherVal else 0

    
    def getHiveDirection(self):
        """getHiveDirection() : returns the relative direction to the hive """
        dx = self.colony.hivePos[0] - self.x
        dy = self.colony.hivePos[1] - self.y
        angle_to_hive = math.atan2(dy, dx)
        relative_angle = angle_to_hive - self.direction
        # Normalize to range [-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        # Normalize to [-1, 1]
        normalized_angle = relative_angle / math.pi
        return normalized_angle

    def getHiveDistance(self):
        """getHiveDistance() : returns the normalized inverse distance to the hive """
        dx = self.colony.hivePos[0] - self.x
        dy = self.colony.hivePos[1] - self.y
        distance = math.sqrt(dx**2 + dy**2)
        max_distance = math.sqrt(self.colony.width**2 + self.colony.height**2)
        # Invert and normalize the distance so that closer distances yield higher values
        normalized_distance = 1 - (distance / max_distance)
        return normalized_distance


    def isCarryingFood(self):
        """isCarryingFood() : returns 1 if the ant is carrying food, else 0 """
        return 1 if self.carryingFood else 0

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
        
        
        ammount*=.5
       
       # ant has no access to the grid, so it cannot check for walls
        # futurePox = [self.x + math.cos(self.direction) * ammount, self.y + math.sin(self.direction) * ammount]
        # curWall = self.wallGrid.GetVal(int(futurePox[0]), int(futurePox[1])
        # if curWall != False:
        #     ammount = 0
        #     self.life -= 1# penalize the ant for hitting a wall
        #     #damage the wall at the front of the ant by .1
        #     self.wallGrid.SetVal(int(futurePox[0]), int(futurePox[1]), curWall - .1)
        
        if self.DebugBrain:
            print(self.blockedFront)
                  
        if self.blockedFront!= False:
            ammount = 0
            self.life -= 1# penalize the ant for hitting a wall
            #damage the wall at the front of the ant by .1
    
        if self.DebugBrain:
            print(f'moving ammount: {ammount}')
    
        addX = math.cos(self.direction) * ammount
        addY = math.sin(self.direction) * ammount
        if self.DebugBrain:
            print(f'addX: {addX}, addY: {addY}')
        
        self.x += addX
        self.y += addY
        if self.DebugBrain:
            print(f'new pos: {self.x}, {self.y}')
        self.energy -= 1
        
        # set farthest traveled if greater than current
        self.FarthestTraveled = max(self.FarthestTraveled, math.sqrt((self.x - self.colony.hivePos[0])**2 + (self.y - self.colony.hivePos[1])**2))
    
    def turn(self, direction):
        """turn() : turn the ant in a direction """
        self.direction += direction
        #limit the direction to 0-2pi
        if self.direction > 2*math.pi:
            self.direction -= 2*math.pi
        if self.direction < -2*math.pi:
            self.direction += 2*math.pi
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
            return False
        if y < 0 or y >= len(self.grid[x]):
            return False
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
        
        self.foodGrid = WorldGrid(GridSize[0], GridSize[1])
        self.wallGrid = WorldGrid(GridSize[0], GridSize[1])
        self.pheromoneGrid = WorldGrid(GridSize[0], GridSize[1])
        self.width = GridSize[0]
        self.height = GridSize[1]

        # self.hivePos = [int(GridSize[0]*0.5), int(GridSize[1]*0.5)]
        self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]
        #the upper right corner is the tricky spot to lets put it there
        # self.hivePos = [int(GridSize[0]*0.1), int(GridSize[1]*0.1)]
        
        self.StartTime = time.time()
        
        self.BestAnts = []
        self.LastBestAnts = []
        
        self.TopPheromone = 0

        self.LastSave = time.time()
        
        self.UpdateTime = 0
        
        self.totalCreatedAnts = 0
        self.totalDeadAnts = 0
        self.topFoodFound = 0
        self.totalSteps = 0
        
        # Battery level tracking for Pi mode
        self.batteryLevel = 0
        self.lastBatteryUpdate = 0
        self.batteryUpdateInterval = 10  # Update every 10 seconds
        
        self.create_world()
        
                 
    def WorldToScreen(self, apos):
        ax = apos[0]/self.width * self.screenSize[0]
        ay = apos[1]/self.height * self.screenSize[1]
        return (ax, ay)

    def add_ant(self, brain=None, startP=None):
        ant = Ant(self)
        antID = [self.totalCreatedAnts, 0, -1]
        self.totalCreatedAnts += 1
        if self.FollowNextAnt:
            ant.DebugBrain = True
            self.FollowNextAnt = False
        
        ant.x = random.randint(0, self.width)
        ant.y = random.randint(0, self.height)
        if startP:
            ant.x = startP[0]
            ant.x += random.random() * 3
            ant.y = startP[1]
            ant.y += random.random() * 3
        
        
        
        if brain:
            ant.brain = brain
        else:
            antID[1] = 'N'
            ant.create_brain(random.randint(6, 64)) #much more complicated ant brains maybe is better?
        ant.antID = antID
        
        #load color
        ant.SetColor()
        
        self.ants.append(ant)
        
        return ant
        # print(f'added new ant with brain {ant.brain}')

    def add_food(self, pos=None):
        """add to the food grid in a random spot"""
        foodX = random.randint(0, self.width)
        foodY = random.randint(0, self.height)

        #make sure it doesnt drop on a wall or near the hive
        worldVal = self.wallGrid.GetVal(foodX, foodY)
        if worldVal == False or worldVal == []:
            distToHive = math.sqrt((foodX - self.hivePos[0])**2 + (foodY - self.hivePos[1])**2)
            if distToHive < 15:
                return

        if pos:
            foodX = pos[0]
            foodY = pos[1]
            
        #SKIP IF OUTSIDE BOUNDS
        if foodX < 0 or foodX >= self.width:
            return
        if foodY < 0 or foodY >= self.height:
            return

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
        # for i in range(1000):
            # self.add_food()
        # make a cross of walls 70 percent of width and height
        startX = int(self.width * 0.15)
        endX = int(self.width * 0.85)
        startY = int(self.height * 0.15)
        endY = int(self.height * 0.85)
        
        # for i in range(startX, endX):
        #     self.wallGrid.SetVal(i, int(self.height/2), 1)
        # for i in range(startY, endY):
        #     self.wallGrid.SetVal(int(self.width/2), i, 1)
            
        
        #just make some random walls around
        #width*height * .3
        numWalls = int(self.width * self.height * .15)
        
            
        # #add some random walls
        for i in range(numWalls):
            self.add_wall()
        
        #create some walls in a ring shape
        # use a random number to decide if the ring is filled or not
        #ring 1
        
         
        # for r in range(0, 3):
            
        #     ringDiameter = (r+2) * 10
            
        #     openingAng = random.randint(0, 360)
            
        #     for angle in range(0, 360):
        #         x = int(self.width/2 + math.cos(math.radians(angle)) * ringDiameter)
        #         y = int(self.height/2 + math.sin(math.radians(angle)) * ringDiameter)
                
        #         #leave a 10degree gap in a random spot
        #         if angle > openingAng and angle < openingAng + 90:
        #             continue
        #         self.wallGrid.SetVal(x, y, 1)
                
           
    def Repopulate(self):
        """repopulate the ants with the best ants"""
        # print("Repopulating")
        # currentAnts = len(self.ants)
        bestAntNum = len(self.BestAnts)
        
        if bestAntNum == 0:
            #just make new ants
            while len(self.ants) < self.maxAnts:
                self.add_ant(brain=None, startP=self.hivePos)
            return {'ants':len(self.ants), 'probbest':0}
        #sort and trim the best ants to top 50 ants
        # self.BestAnts = sorted(self.BestAnts, key=lambda x: x["food"], reverse=True)
    
        #sort by fitness now, highest fitness first
        self.BestAnts = sorted(self.BestAnts, key=lambda x: x["fitness"], reverse=True)
        
        self.BestAnts = self.BestAnts[:100] #only keep the top 10 ants
        probBest = .1
        
        # print(f'Best Ants: {self.BestAnts}')
        
        #create new ants from the best ants
        whileCount = 0
        while len(self.ants) < self.maxAnts:
            whileCount += 1
            if whileCount > 1000:
                print("Stuck in while loop")
                break
            
            # ant = Ant()
            bestAntScore = self.BestAnts[0]["food"]
            
            
            
            #make more new ants if the best ant score is less than 10
            
            # if bestAntScore < 30:
            #     probBest = 1-(bestAntScore / 60) # the higher the score, the more likely we add a best ant
            if bestAntScore < 2:
                probBest = .9
            elif bestAntScore < 5:
                probBest = .2
            else:
                probBest = 0
            
            if random.random() < probBest: # if less then probBest, add a random ant
                self.add_ant(brain=None, startP=self.hivePos)
                print("Adding random ant")
            else: # add a best ant or pick two best ants as parents
                if random.random() > 0.5: #fifty fifty chance to mutate the brain or parent
                    # print("Mutating")
                    randomInt = random.randint(0, len(self.BestAnts)-1)
                    # randomInt = random.randint(0, min(5, len(self.BestAnts)-1)) #only pick the top 5 ants
                
                    # this randInt is also a fractional of the length of the best ants
                    pickedFraction = randomInt / len(self.BestAnts)
                    antBonus = 1 - pickedFraction # the higher the fraction, the more likely we add a best ant
                    antBonus = int(antBonus * 10) #scale the bonus
                    antBonus = antBonus * antBonus #square the bonus
                    antBonus = max(1, antBonus) #make sure we add at least one ant
                    #fitness
                    aFit = self.BestAnts[randomInt]["fitness"]
                    # print(f'Fitness: {aFit}, AntBonus: {antBonus}')
                    
                    bestPick = self.BestAnts[randomInt]
                    newBrain = bestPick["brain"].copy()
                    
                    newType = "CL" #clone
                    cloneID = -1
                    
                    if random.random() > 0.5: #mutate the brain or not
                        newType = "M" #mutate
                        newBrain = self.MutateBrain(newBrain)
                    # elif random.random() > 0.5: #shuffle the brain
                    #     newType = "S"
                    #     random.shuffle(newBrain) #shuffle the brain sequence so that the brain fires in a different order
                    else:
                        cloneID = bestPick["antID"][0]
                    
                    # print(f'Adding {antBonus} new ants, type : {newType}')
                    for i in range(antBonus): #ants higher in the list get more clones
                        newAnt = self.add_ant(brain=newBrain, startP=self.hivePos)
                        newAnt.antID[1] = newType# keep track of what type of ant it is
                        newAnt.antID[2] = cloneID # keep track of the clone id, the id of the ant that was cloned

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
                    # print(f'Adding new ant from parents')
                    newAnt = self.add_ant(brain=newBrain, startP=self.hivePos)
                    newAnt.antID[1] = "CH" #child
        return {'ants':len(self.ants), 'probbest':probBest}            
    
    def ReplenishFood(self, quadrant, ammt):
        """add food to the grid"""
        # print("replenishing food")
        # print(f'quadrant: {quadrant}')
        # specific quadrant for each food every minute
        timeSinceStart = time.time() - self.StartTime
        #every 1 minute switch the food to a new quadrant
        # quadrant = int(timeSinceStart / 240) % 4

        quads = []
        for i in range(2):
            for j in range(2):
                quad = [[i*self.width/2, j*self.height/2], [i*self.width/2, (j+1)*self.height/2], [(i+1)*self.width/2, j*self.height/2], [(i+1)*self.width/2, (j+1)*self.height/2]]
                quads.append(quad)
        # count current food and add more as needed
        currentFood = len(self.foodGrid.listActive())
        
        whileCount = 0
        while currentFood < ammt: # food scaricity produces more competition for food
            # print(f'Current Food: {currentFood} Ammt: {ammt}')
            whileCount += 1
            if whileCount > 100:
                print("Stuck in while loop")
                #PICK A NEW QUADRANT
                quadrant = random.randint(0, 3)
                break
           
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
            for i in range(60):
                foodPosRand = [foodPos[0] + random.randint(-5, 5), foodPos[1] + random.randint(-5, 5)]
                #dont drop within 20 tiles of the hive
                #MUST BE WITHIN BOUNDS OF GRID
                if foodPosRand[0] < 0 or foodPosRand[0] >= self.width:
                    continue
                if foodPosRand[1] < 0 or foodPosRand[1] >= self.height:
                    continue
                distToHive = math.sqrt((foodPosRand[0] - self.hivePos[0])**2 + (foodPosRand[1] - self.hivePos[1])**2)
               
                if distToHive > 25:
                    # print(f'distToHive: {distToHive}')
                    #dont drop on a wall
                    worldVal = self.wallGrid.GetVal(foodPosRand[0], foodPosRand[1])
                    # print (f'worldVal: {worldVal}')
                    if worldVal == False or worldVal == []:
                        self.add_food(foodPosRand)
                        currentFood = len(self.foodGrid.listActive())
        # print('food replenished')


    def MutateBrain(self, brain):
        """mutate the brain by changing one of the values"""
        if len(self.ants) == 0:
            return brain
        numChanges = random.randint(1, 4)
        for i in range(numChanges):
            idx = random.randint(0, len(brain)-1)
            src, selectorSrc, selectorDst, frc, dest = brain[idx]
            change = random.randint(0, 4)
            if change == 0:
                src = not src
                # Re-assign selectorSrc based on the new src value
                if src:
                    selectorSrc = random.randint(0, len(self.ants[0].InputSources) - 1)
                else:
                    selectorSrc = random.randint(0, len(self.ants[0].neurons) - 1)
            elif change == 1:
                # Change selectorSrc within the appropriate range
                if src:
                    selectorSrc = random.randint(0, len(self.ants[0].InputSources) - 1)
                else:
                    selectorSrc = random.randint(0, len(self.ants[0].neurons) - 1)
            elif change == 2:
                # Change selectorDst within the appropriate range
                if dest:
                    selectorDst = random.randint(0, len(self.ants[0].OutputDestinations) - 1)
                else:
                    selectorDst = random.randint(0, len(self.ants[0].neurons) - 1)
            elif change == 3:
                frc = random.uniform(-1, 1)
            elif change == 4:
                dest = not dest
                # Re-assign selectorDst based on the new dest value
                if dest:
                    selectorDst = random.randint(0, len(self.ants[0].OutputDestinations) - 1)
                else:
                    selectorDst = random.randint(0, len(self.ants[0].neurons) - 1)
            brain[idx] = (src, selectorSrc, selectorDst, frc, dest)
        return brain


    def isAtHive(self, ant):
        hiveRadius = 5
        distToHive = math.sqrt((ant.x - self.hivePos[0])**2 + (ant.y - self.hivePos[1])**2)
        return distToHive < hiveRadius
            
    def update(self):
        self.totalSteps += 1
        # print("UPDATE SIM")
        
        # Update battery level for Pi mode every 10 seconds
        current_time = time.time()
        if PISUGAR_AVAILABLE and (current_time - self.lastBatteryUpdate) >= self.batteryUpdateInterval:
            try:
                self.batteryLevel = pisugar.get_battery_level()
                self.lastBatteryUpdate = current_time
            except Exception as e:
                print(f"Error getting battery level: {e}")
                self.batteryLevel = 0
        
        if self.totalSteps % 100 == 0:
            print(f'Current Steps: {self.totalSteps}')

        #number of ants
        # print(f'Number of ants: {len(self.ants)}')
           
        startTime = time.time()
        for ant in self.ants:
            # print(f'Ant: {ant.antID}')
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
            if self.wallGrid.GetVal(frontPos[0], frontPos[1]) != []:
                # print(self.wallGrid.GetVal(frontPos[0], frontPos[1]) )
                ant.blockedFront = [frontPos[0], frontPos[1]]
            else:
                ant.blockedFront = False
        
            if frontPos[0] < 0 or frontPos[0] >= self.width:
                ant.blockedFront = True
            if frontPos[1] < 0 or frontPos[1] >= self.height:
                ant.blockedFront = True
            
            # # find the closest food
            # closest = 1000000
            closestFood = [-1,-1]
            # for food in self.foodGrid.listActive():
            #     # dist = math.sqrt((ant.x - food[0])**2 + (ant.y - food[1])**2)
            #     dist = (ant.x - food[0])**2 + (ant.y - food[1])**2

            #     if dist < closest:
            #         closest = dist
            #         if dist < 10:
            #             closestFood = food
                        
            #alternatively, we can search in a spiral pattern up to a distance of 10 tiles
            #for example :
            beenHere = []
            aX = int(ant.x)
            aY = int(ant.y)
            for i in range(1, 10):
                for j in range(-i, i):
                    for k in range(-i, i):
                        #check if we have been here
                        if [aX + j, aY + k] in beenHere:
                            continue
                        beenHere.append([aX + j, aY + k])
                        if self.foodGrid.GetVal(aX + j, aY + k) != []:
                            closestFood = [aX + j, aY + k]
                            break
                    if closestFood != [-1,-1]:
                        break
                if closestFood != [-1,-1]:
                    break
                      
                      
            if closestFood[0] < 0 or closestFood[0] >= self.width:
                    closestFood = [-1,-1]
            if closestFood[1] < 0 or closestFood[1] >= self.height:
                    closestFood = [-1,-1]
                      
                      
            if closestFood != [-1,-1]:
                ant.ClossestFood = closestFood

            if self.isAtHive(ant):
                if ant.carryingFood:
                    ant.carryingFood = False
                    # Reward the ant
                    ant.life += 100
                    ant.fitness += 100 #extra reward for returning food home
                    # print(f'Ant {ant.antID} returned food to the hive!!')
                
            ant.RunBrain()
            ant.pDirection = float(ant.direction)
           
           
           
           ### history saving
        #     lastknownPos = ant.posHistory[-1] if len(ant.posHistory) > 0 else [0,0]
     
        #     moveDist = math.sqrt((ant.x - lastknownPos[0])**2 + (ant.y - lastknownPos[1])**2)
        # # limited to 1 tile per move to prevent too much history
        #     if abs(moveDist) > 1:
        #         ant.posHistory.append([ant.x, ant.y])
           ### end history saving

            ant.life -= 1
            if ant.life <= 0: #DEAD ANT
                #if debugbrain, pick another ant
                if ant.DebugBrain:
                    self.FollowNextAnt = True
                self.ants.remove(ant)
                #check how much food the ant consumed
                foodConsumed = ant.FoodConsumed
                antFitness = ant.fitness
                
                foodBonus = foodConsumed *10 #bonus for food consumed 
                antFitness += foodBonus
                
                # bonus for distance traveled from the hive while alive
                distanceBonus = ant.FarthestTraveled * 0.05
                if distanceBonus < 1:
                    distanceBonus = 0
                distanceBonus = int(distanceBonus**4)
                
                #ONLY IF EAT ONE FOOD ATLEAST
                if foodConsumed > 0:
                    antFitness += distanceBonus
                # if distanceBonus > 0:
                    # print(f"dead ant distanceBonus: {distanceBonus}")
                antBrain = ant.brain
                self.totalDeadAnts += 1
                if antFitness > 2:
                    self.BestAnts.append({"food":foodConsumed, "brain":antBrain, "antID":ant.antID, "fitness":antFitness})
            # topFoodCount = self.BestAnts[0]["food"] if len(self.BestAnts) > 0 else 0
            
            #check if the ant has a closestFood value, if so detect the distance and if close enough, consume the food
            
            antClosestFood = ant.ClossestFood
            if antClosestFood != [-1,-1]:
                distToFood = math.sqrt((ant.x - antClosestFood[0])**2 + (ant.y - antClosestFood[1])**2)
                if distToFood < 1: #give a little leeway
                    # if ant.carryingFood == False: #if the ant is not carrying food force ants to return home to keep eating
 
                    if self.foodGrid.RemoveVal(antClosestFood[0], antClosestFood[1]):
                    # print(f'removing food at {antClosestFood}')
                        ant.ClossestFood = [-1,-1]
                        ant.energy += 10
                        ant.FoodConsumed += 1
                        ant.life += 50 #reward the ant for finding food
                        ant.fitness += 1
                        if ant.life > 200:
                            ant.life = 200 #keep things reasonable, some ants are too good

                        ant.carryingFood = True

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

        # print('ants updated')
        #remove old pherimone cells, anything less than 0
        activePhers = self.pheromoneGrid.listActive()
        # print(f'activePhers: {len(activePhers)}')
        for pher in activePhers:
            # print(f'pher: {pher}')
            if pher[2] <= 0:
                self.pheromoneGrid.RemoveVal(pher[0], pher[1])
            else:
                #decay the pheromone
                self.pheromoneGrid.SetVal(pher[0], pher[1], pher[2]-.005)

        # print('pheromone updated')
        quadrant = int(self.totalSteps / 200) % 4

        self.ReplenishFood(quadrant, 250)
        # print('food replenished')

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
            if "probbest" in repop_result:
                print(f'Repopulate Probiability of random ant: {repop_result["probbest"]}')
            print("----------------------------------------")
            #randomize the hive pos
            # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]


        if self.totalSteps % 10000 == 0:
            self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]

        # print('update done')


    def LoadBestAnts(self, quantity):
        """load the best ants from a file"""
        print("Loading Best Ants")
        bestAntsFound = []
        #find and load every file you can find. add all ants and select the top 50
        
        searchFolder = 'dataSave'
        for root, dirs, files in os.walk(searchFolder):
            for file in files:
                if file.endswith('.json'):
                    with open(f'{searchFolder}/{file}', 'r') as f:
                        print(f'Loading: {file}')
                        data = f.read()
                        data = json.loads(data)
                        bestAnts = data["BestAnts"]
                        bestAntsFound.extend(bestAnts)
        # Handle missing fitness keys in old data files
        for ant in bestAntsFound:
            if "fitness" not in ant:
                # Calculate fitness from food for old data files
                ant["fitness"] = ant["food"] * 10  # Basic conversion from food to fitness
        
        #sort the best ants
        # bestAntsFound = sorted(bestAntsFound, key=lambda x: x["food"], reverse=True)
        bestAntsFound = sorted(bestAntsFound, key=lambda x: x.get("fitness", x["food"] * 10), reverse=True)
        #filter and make sure unique ant brains only
        # bestAntsFound = [dict(t) for t in {tuple(d.items()) for d in bestAntsFound}]
        brainList = []
        freshAnts = []
        for ant in bestAntsFound:
            if ant["brain"] not in brainList:
                brainList.append(ant["brain"])
                freshAnts.append(ant)
        bestAntsFound = freshAnts

        #randomize best ants
        print(f'Found {len(bestAntsFound)} best ants from files')
        if len(bestAntsFound) > 100:
            numTopAnts = min(500, len(bestAntsFound))
            bestAntsFound = bestAntsFound[:numTopAnts] #only keep the top 500 ants
            print(f"Best Ant Range: {bestAntsFound[0].get('fitness', 'N/A')} - {bestAntsFound[-1].get('fitness', 'N/A')}")
            
            random.shuffle(bestAntsFound)
            #add these ants to the game
            
            #top ants getmore new ants

            loadedAnts = 0
            for i in range(int(quantity)):
                randomPick = random.randint(0, len(bestAntsFound)-1)
                antBrain = bestAntsFound[randomPick]["brain"].copy()
                # print(f'antBrain: {antBrain}')
                # print(f'antBrainLen: {len(antBrain[0])}')
                if len(antBrain[0])==5: ## old version onlyhad 3 values
                    NAnt = self.add_ant(brain=antBrain, startP=self.hivePos)
                    NAnt.antID[1] = 'L' #loaded ant
                    loadedAnts += 1
                
            print(f'Loaded {loadedAnts} best ants from file')


    def drawPaths(self, screen, isPi=False):
        # fill screen with .9 alpha so history fades away
        # screen.setAlpha(10)
        # screen.fill((25, 25, 25, 50), special_flags=pygame.BLEND_RGBA_MULT)
        
        fade = pygame.Surface((self.screenSize[0], self.screenSize[1]))
        fade.fill((25, 25, 25))
        # if isPi:
        #     fade.set_alpha(10)
        # else:
        #     fade.set_alpha(5)
        fade.set_alpha(5)
        screen.blit(fade, (0, 0))
        if isPi:
            for ant in self.ants:
                if ant.life <= 1:
                    if len(ant.posHistory) > 1:
                        #ant must have eaten one food
                        if ant.FoodConsumed > 0:
                            for i in range(len(ant.posHistory) - 1):
                                p1 = self.WorldToScreen(ant.posHistory[i])
                                p2 = self.WorldToScreen(ant.posHistory[i + 1])
                                pygame.draw.line(screen, ant.Color, p1, p2, 1)
                                
        else:
            for ant in self.ants:
                if len(ant.posHistory) > 1:
                    #ant must have eaten one food
                    if ant.FoodConsumed > 0:
                        for i in range(len(ant.posHistory) - 1):
                            p1 = self.WorldToScreen(ant.posHistory[i])
                            p2 = self.WorldToScreen(ant.posHistory[i + 1])
                            pygame.draw.line(screen, ant.Color, p1, p2, 1)
                        
        timeDelta = time.time() - self.LastSave
        timeDistance = 60 #save every minute
      


        if timeDelta > timeDistance:
            #save the best ants to a file
            self.saveData()
            #also save an image of the screen\
            if isPi == False:
                tCode = time.strftime("%Y%m%d-%H%M%S")
                pygame.image.save(screen, f'dataSave/{tCode}.png')
            
            self.LastSave = time.time()



    def drawAnts(self,screen, isPi=False):
        screen.fill((25,25,25))
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
            pygame.draw.circle(screen, (0, 200, 0), (int(cx), int(cy)), int(self.TileSize/2), 1)
            
        for wall in self.wallGrid.listActive():
            wpxy = self.WorldToScreen(wall)
            wpxy = (int(wpxy[0]), int(wpxy[1]))
            # pygame.draw.rect(screen, (100,100,100), (wpxy[0], wpxy[1], self.TileSize, self.TileSize))
            # small gray hollow box with x inside
            pygame.draw.rect(screen, (100,100,100), (wpxy[0], wpxy[1], self.TileSize, self.TileSize), 1)
            # pygame.draw.line(screen, (100,100,100), (wpxy[0], wpxy[1]), (wpxy[0]+self.TileSize-1, wpxy[1]+self.TileSize-1))
            # pygame.draw.line(screen, (100,100,100), (wpxy[0]+self.TileSize, wpxy[1]), (wpxy[0]-1, wpxy[1]+self.TileSize-1))
            

        for ant in self.ants:
            pxy = self.WorldToScreen((ant.x, ant.y))
            pxy = (int(pxy[0]), int(pxy[1]))
            #square if the ant is blocked
            
            RV = ant.Color[0]
            GV = ant.Color[1]
            BV = ant.Color[2]
                            
            #if ant.blockedFront:
            # pygame.draw.rect(screen, (RV,GV,BV), (pxy[0]-.5, pxy[1]-.5, 1, 1))
            #a single pixel is like this : surface.set_at((x, y), color)
            screen.set_at((pxy[0], pxy[1]), (RV,GV,BV))

            #if ant has food, make a box around ant
            if ant.carryingFood:
                pygame.draw.rect(screen, (RV,GV,BV), (pxy[0]-2, pxy[1]-2, 5, 5), 1)

            #pos
            # print(f'ant pos: {pxy}')
            # else:
                # pygame.draw.circle(screen, (RV,GV,BV), (int(pxy[0]), int(pxy[1])), 2.5)
                
            #draw the path of the ant
            # print(ant.posHistory)
          
          

            #draw the direction of the ant with a line
            # pygame.draw.line(screen, (0, 0, 0), (pxy[0], pxy[1]), (pxy[0] + math.cos(ant.direction) * 10, pxy[1] + math.sin(ant.direction) * 10))



            #debug show a line to the closest food
            # if ant.ClossestFood != [-1,-1]:
            #     fpxy = self.WorldToScreen(ant.ClossestFood)
            #     pygame.draw.line(screen, (0, 255, 0), (pxy[0], pxy[1]), (fpxy[0], fpxy[1]))
            if ant.DebugBrain:
                
                #draw a hollow arount the ant 20 pixels
                
                #delay for 10 milliseconds
                pygame.time.delay(50)
                
                pygame.draw.circle(screen, (RV,GV,BV), (int(pxy[0]), int(pxy[1])) , 20, width=1)
                
                #draw the direction of the ant with a line
                pygame.draw.line(screen, (0, 0, 0), (pxy[0], pxy[1]), (pxy[0] + math.cos(ant.direction) * 20, pxy[1] + math.sin(ant.direction) * 20 ))
                
                #show the ant info next to the ant
                #direction
                font = pygame.font.Font(None, 20)
                text = font.render(f'DIR:{ant.direction}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]))
                #life
                text = font.render(f'LIFE:{ant.life}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+15))
                #blockedFront
                text = font.render(f'BLOCKED:{ant.blockedFront}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+30))
                #closest food
                text = font.render(f'CLOSEST F:{ant.ClossestFood}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+45))
                #foodfront
                text = font.render(f'PHER F:{ant.pherFront}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+60))
                #close to food
                text = font.render(f'CLOSE To FOOD:{ant.closeToFood()}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+75))
                #foodDir
                text = font.render(f'FOOD DIR:{ant.foodDir()}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+90))
                
                #draw direction to food with line
                if ant.ClossestFood != [-1,-1]:
                    fpxy = self.WorldToScreen(ant.ClossestFood)
                    pygame.draw.line(screen, (0, 255, 0), (pxy[0], pxy[1]), (fpxy[0], fpxy[1]))
                
                
                
                
                #display the brain in a box in the right bottom corner
                #make three columns, one for all the optional inputs, one for the neurons and one for the outputs
                
                
                #draw brain drawbrain
            
                #draw rect box, gray bg
                pygame.draw.rect(screen, (50,50,50), (self.screenSize[0]-300, self.screenSize[1]-300, 300, 300))
                #draw circles for the inputs
                column_1 = 50
                column_2 = 150
                column_3 = 250
                
                offsetX = self.screenSize[0]
                
                for i in range(len(ant.InputSources)):
                    pygame.draw.circle(screen, (255, 255, 255), (offsetX-column_3, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the input in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.InputSources[i]()}', True, (255,255,255))
                    screen.blit(text, (offsetX-column_3, self.screenSize[1]-250 + i*20))
                    
                    
                for i in range(len(ant.neurons)):
                    pygame.draw.circle(screen, (255, 255, 255), (offsetX-column_2, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the neuron in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.neurons[i]}', True, (255,255,255))
                    screen.blit(text, (offsetX-column_2, self.screenSize[1]-250 + i*20))
                    
                
                for i in range(len(ant.OutputDestinations)):
                    pygame.draw.circle(screen, (255, 255, 255), (offsetX-column_1, self.screenSize[1]-250 + i*20), 5)
                    #put the value of the output in the circle
                    font = pygame.font.Font(None, 20)
                    text = font.render(f'{ant.OutputDestinations[i]}', True, (255,255,255))
                    screen.blit(text, (offsetX-column_1, self.screenSize[1]-250 + i*20))
                    
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
                        xStart = column_3
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
                        xEnd = column_1
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
                        
                    xStart = offsetX - xStart
                    xEnd = offsetX - xEnd
                    
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
        # dataShow = self.BestAnts[:30]
        dataShow = self.BestAnts
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
            cloneParent = ant["antID"][2] if ant["antID"][2] != -1 else ''
            #type to color
            ColorLookup = {"CL":(200, 0, 200), "M":(255, 200, 0), "CH":(0, 200, 0), "N":(255, 0, 0)}
            # ad fitness to the string
            textV = f'Food: {int(ant["food"]):03} Fitness: {int(ant["fitness"]):03} ID: {ant["antID"][0]:08}, {ant["antID"][1]}, {cloneParent}'
            # find the ant color and make a small colored box next to the text
            #ant color fromes from the brain itself
            
            antColor = BrainToColor(ant["brain"])
            
            antColor = (antColor[0], antColor[1], antColor[2])
            
            #color white
            #clip the text to 50 characters
            textV = textV[:80]
            
            # font = pygame.font.Font(None, 20)
            #monospace font
            font = pygame.font.SysFont('monospace', 12)
            # text = font.render(textV, True, (255,255,255))
            #set color based on the type of ant
            antType = ant["antID"][1]
            color = (255, 255, 255)
            if antType in ColorLookup:
                color = ColorLookup[antType]
            text = font.render(textV, True, color)
            screen.blit(text, (30, 10 + i*12))
            pygame.draw.rect(screen, antColor, (10, 10 + i*12, 10, 10))
            
        # Display battery level for Pi mode (50px line at bottom)
        if isPi and PISUGAR_AVAILABLE:
            battery_bar_height = 50
            battery_bar_y = self.screenSize[1] - battery_bar_height
            battery_bar_width = self.screenSize[0]
            
            # Background bar (dark gray)
            pygame.draw.rect(screen, (50, 50, 50), (0, battery_bar_y, battery_bar_width, battery_bar_height))
            
            # Battery level bar (green to red gradient based on level)
            if self.batteryLevel > 0:
                battery_width = int((self.batteryLevel / 100.0) * battery_bar_width)
                
                # Color based on battery level
                if self.batteryLevel > 50:
                    battery_color = (0, 255, 0)  # Green
                elif self.batteryLevel > 20:
                    battery_color = (255, 255, 0)  # Yellow
                else:
                    battery_color = (255, 0, 0)  # Red
                
                pygame.draw.rect(screen, battery_color, (0, battery_bar_y, battery_width, battery_bar_height))
                
                # Battery percentage text
                font = pygame.font.Font(None, 36)
                battery_text = f"Battery: {self.batteryLevel}%"
                text_surface = font.render(battery_text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(battery_bar_width // 2, battery_bar_y + battery_bar_height // 2))
                screen.blit(text_surface, text_rect)
        
    def saveData(self):
        
        #check to see if the data chenged from last time
        if self.BestAnts == self.LastBestAnts:
            # print('No changes to save')
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
        self.drawPaths = False
        self.debugMode = False
        
        self.maxAnts = 500
        
        self.screenSize = (1000, 1000)
        #using argparse
        parser = argparse.ArgumentParser(description='Run the ant simulation')
        parser.add_argument('--pi', action='store_true', help='Run on a Raspberry Pi')
        parser.add_argument('--paths', action='store_true', help='Draw the paths of the ants')
        #add debug mode
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        args = parser.parse_args()
        if args.pi:
            self.isPi = True
            print('Running on Raspberry Pi')
            self.screenSize = (480, 1920)

        if args.paths:
            self.drawPaths = True
            print('Path Mode')

        if args.debug:
            print('Debug Mode')
            self.debugMode = True

        
        # if len(sys.argv) > 1:
        #     print('Arguments: ', sys.argv[1])
        #     if sys.argv[1] == "pi":
        #         isPi = True
        #         print('Running on Raspberry Pi')

        pygame.init()
        
        tileSize = 15
                
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

            self.maxAnts = 80
            tileSize = 25
        
        else:
            # os.environ["SDL_VIDEO_WINDOW_POS"] = "-1100,0"
            self.screen = pygame.display.set_mode(self.screenSize)
            
            
        self.clock = pygame.time.Clock()

           
        print('Creating Ant Colony')
        self.antColony = AntColony(self.screenSize, self.maxAnts, tileSize)
        
        self.antColony.LoadBestAnts( self.maxAnts )

        #first run update 20000 times
        lastPercent = 0
        if self.isPi == False:
            newAnts = 500
            #add new ants before training
            print(f'Adding {newAnts} new random ants')
            for i in range(newAnts):
                self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            numRuns = 0
            maxFoodFound = 0
            print('Pre-Training Ants')
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
        print("Game Ready")
    
    def run(self):
        
        running = True
        print('Running PYGAME instance now')
        ticks = 0
        while running:

     
            ticks += 1

            #dubug only
            if self.debugMode:
                if ticks % 10 == 0:
                    print(f'Ticks: {ticks}')
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            #run 5 times between each draw
            for i in range(5):
                self.antColony.update()


            fps = self.clock.get_fps()
           

            if self.drawPaths:
            # self.antColony.drawAnts(self.screen, isPi=self.isPi)
                self.antColony.drawPaths(self.screen, isPi= self.isPi)
            else:
                self.antColony.drawAnts(self.screen, isPi=self.isPi)
                text = f'FPS: {fps}'
                font = pygame.font.Font(None, 26)
                text = font.render(text, True, (255, 255, 255))
                self.screen.blit(text, (self.screenSize[0]-100, 10))
                #num ants displayed
                text = f'Ants: {len(self.antColony.ants)}'
                font = pygame.font.Font(None, 26)
                text = font.render(text, True, (255, 255, 255))
                self.screen.blit(text, (self.screenSize[0]-100, 30))



            # KEY PRESSES
            keys = pygame.key.get_pressed()

            # Quit application with Q key or ESC key
            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                print("Quitting application...")
                running = False

            # Shutdown raspberry pi with X key (only in pi mode)
            if keys[pygame.K_x] and self.isPi:
                print("Shutting down Raspberry Pi...")
                self.antColony.saveData()  # Save data before shutdown
                pygame.quit()
                os.system("sudo shutdown -h now")
                return

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

            # if you hit key R, add 1000 random ants
            if keys[pygame.K_r]:
                for i in range(10000):
                    self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            #if p is pressed, pick the next ant to debug the brain
            if keys[pygame.K_p]:
                # for ant in self.antColony.ants:
                #     ant.DebugBrain = False
                #     break
                self.antColony.FollowNextAnt = True
                
            #add key will add 1000 ants
            if keys[pygame.K_a]:
                #make max ants 1000
                self.maxAnts += 1000
                self.antColony.maxAnts = self.maxAnts
                # for i in range(1000):
                #     self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            

            ###
            
            

            if fps < 2:
                self.maxAnts-=1
                #also change the antcolonys max ants
                self.antColony.maxAnts = self.maxAnts
            elif fps > 10:
                self.maxAnts+=1
                self.antColony.maxAnts = self.maxAnts
                
            if self.maxAnts < 300:
                self.maxAnts = 300
                self.antColony.maxAnts = self.maxAnts

            pygame.display.flip()
            # self.clock.tick(120)
            self.clock.tick()

        pygame.quit()

if __name__ == "__main__":
    print('Starting Simulation')
    game = Game()
    game.run()
