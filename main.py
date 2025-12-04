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
        # self.energy = 100
        self.life = 200
        self.FoodConsumed = 0
        self.blockedFront = False
        self.ClossestFood = [-1,-1]

        self.carryingFood = False
        
        # Navigation fitness tracking
        self.foodPickupPos = None  # Store position where food was picked up
        self.navigationFitness = 0  # Store calculated navigation fitness
        
        self.FarthestTraveled = 0 # the farthest the ant has traveled from the hive give a fitness bonus when dead
        
        # Track previous cell position to only drop pheromones when moving to new cell
        self.prevCellX = -1
        self.prevCellY = -1


        #self.InputSources = [ self.getDirection,self.getPdirection, self.getBlockedFront, self.foodDir, self.foodFront, self.oscillate, self.randomNum, self.GetPherFront, self.closeToFood ]
        self.InputSources = [
            self.getDirection, self.getPdirection, self.getBlockedFront,
            self.foodDir, self.foodFront, self.oscillate, self.randomNum,
            self.getNestPherFront, self.getFoodPherFront, self.closeToFood,
            self.getNestPherLeft, self.getNestPherRight, self.getNestPherBack,
            self.getFoodPherLeft, self.getFoodPherRight, self.getFoodPherBack,
            self.getHiveDirection, self.getHiveDistance,
            self.isCarryingFood  # Add here
        ]

       
        self.OutputDestinations = [ self.move, self.turn ]

        # # Dual pheromone tracking
        # self.nestPherFront = 0
        # self.foodPherFront = 0
        
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
           
    def calculateNavigationFitness(self, currentPos=None):
        """Calculate navigation fitness based on food pickup location and current position"""
        if self.foodPickupPos is None:
            return 0
        
        # Use current ant position if not provided
        if currentPos is None:
            currentPos = [self.x, self.y]
        
        # Calculate distances
        pickup_to_nest_dist = math.sqrt((self.foodPickupPos[0] - self.colony.hivePos[0])**2 + 
                                      (self.foodPickupPos[1] - self.colony.hivePos[1])**2)
        current_to_nest_dist = math.sqrt((currentPos[0] - self.colony.hivePos[0])**2 + 
                                       (currentPos[1] - self.colony.hivePos[1])**2)
        
        pickup_to_current_dist = math.sqrt((self.foodPickupPos[0] - currentPos[0])**2 + 
                                         (self.foodPickupPos[1] - currentPos[1])**2)

        # Base fitness: 10 points per grid tile from pickup to nest
        # base_fitness = int(pickup_to_nest_dist * 10)
        
        # Bonus/penalty based on progress toward nest
        if current_to_nest_dist < pickup_to_nest_dist:
            # Ant got closer to nest - keep full fitness
            return int(pickup_to_current_dist * 10)
        else:
            # Ant is farther from nest than when it picked up food - subtract fitness
            return -int(pickup_to_current_dist * 10)
    
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
                
                if self.DebugBrain:
                    print(f'Neuron Input: {idx}  value: {val}')
                
                # Use tanh activation function for better neural network behavior
                val = math.tanh(val)


            
            if self.DebugBrain:
                print(f'val*frc: {val * frc}')


            # Normalize force value to prevent extreme outputs
            normalized_force = math.tanh(frc)
            output_value = val * normalized_force
            
            if dest: # if the destination is an output
                # set the value to the output destination
                outIdx = selectorDst % len(self.OutputDestinations)
                self.OutputDestinations[outIdx](output_value)
                if self.DebugBrain:
                    print(f'Output: {self.OutputDestinations[outIdx].__doc__} value: {output_value}')
            else: # if the destination is a neuron
                # set the value to the neuron with proper bounds
                outIdx = selectorDst % len(self.neurons)
                self.neurons[outIdx] += output_value
                
                # Clamp neuron values to prevent overflow/underflow
                if self.neurons[outIdx] > 10.0:
                    self.neurons[outIdx] = 10.0
                elif self.neurons[outIdx] < -10.0:
                    self.neurons[outIdx] = -10.0
                    
                if self.DebugBrain:
                    print(f'Neuron Output {outIdx} Set to: value: {self.neurons[outIdx]}')
            if self.DebugBrain:
                print('____')
        # self.energy -= 1
        if self.DebugBrain:
            print('-------------------------')
        
        # Apply neuron decay to prevent indefinite accumulation
        for i in range(len(self.neurons)):
            self.neurons[i] *= 0.9  # 10% decay per brain cycle


    def randomNum(self):
        """randomNum() : returns a random number between -1 and 1"""
        return random.random() * 2 - 1
    

    def oscillate(self):
        """oscillate() : returns a value between -1 and 1"""
        return math.sin(self.life/10)
    
    # def getDirection(self):
    #     """getDirection() : returns the direction of the ant in -1 to 1 """
    #     outDir = self.direction % (2*math.pi) / (2*math.pi)
    #     return outDir

    # def getPdirection(self):
    #     """getPdirection() : returns the previous direction of the ant in -1 to 1 """
    #     return self.pDirection / (2*math.pi)
    
    def getDirection(self):
        a = (self.direction + math.pi) % (2*math.pi) - math.pi   # [-pi,pi]
        return a / math.pi                                       # [-1,1]

    def getPdirection(self):
        a = (self.pDirection + math.pi) % (2*math.pi) - math.pi
        return a / math.pi

    # def getEnergy(self):
    #     """getEnergy() : returns the energy of the ant """
    #     return self.energy
    
    def getLife(self):
        """getLife() : returns the life of the ant """
        return self.life

    def getBlockedFront(self):
        """getBlockedFront() : returns true if the ant is blocked in front of it """
        # figure out the direction of the ant and the tile in front of it
        # if the tile in front of it is a wall return true
        return 1 if self.blockedFront else 0
     
    def _sense_line(self, grid, steps=3, step=1.0, angle=0.0):
        """Sense pheromone values along a line in a given direction"""
        acc = 0.0
        for i in range(1, steps+1):
            ax = int(self.x + math.cos(self.direction + angle) * (i*step))
            ay = int(self.y + math.sin(self.direction + angle) * (i*step))
            v = grid.GetVal(ax, ay)
            if v not in ([], False, None):
                acc += v
        return acc / steps

    # Nest pheromone sensors (dropped by ants NOT carrying food - paths to nest)
    def getNestPherFront(self):
        """getNestPherFront() : returns the amount of nest pheromone in front of the ant """
        return self._sense_line(self.colony.nestPheromoneGrid, steps=3, step=1.0, angle=0.0)
    
    def getNestPherLeft(self):
        """getNestPherLeft() : returns the amount of nest pheromone to the left of the ant """
        return self._sense_line(self.colony.nestPheromoneGrid, steps=3, step=1.0, angle=-math.pi/2)

    def getNestPherRight(self):
        """getNestPherRight() : returns the amount of nest pheromone to the right of the ant """
        return self._sense_line(self.colony.nestPheromoneGrid, steps=3, step=1.0, angle=math.pi/2)

    def getNestPherBack(self):
        """getNestPherBack() : returns the amount of nest pheromone behind the ant """
        return self._sense_line(self.colony.nestPheromoneGrid, steps=2, step=1.0, angle=math.pi)

    # Food pheromone sensors (dropped by ants carrying food - paths to food)
    def getFoodPherFront(self):
        """getFoodPherFront() : returns the amount of food pheromone in front of the ant """
        return self._sense_line(self.colony.foodPheromoneGrid, steps=3, step=1.0, angle=0.0)
    
    def getFoodPherLeft(self):
        """getFoodPherLeft() : returns the amount of food pheromone to the left of the ant """
        return self._sense_line(self.colony.foodPheromoneGrid, steps=3, step=1.0, angle=-math.pi/2)

    def getFoodPherRight(self):
        """getFoodPherRight() : returns the amount of food pheromone to the right of the ant """
        return self._sense_line(self.colony.foodPheromoneGrid, steps=3, step=1.0, angle=math.pi/2)

    def getFoodPherBack(self):
        """getFoodPherBack() : returns the amount of food pheromone behind the ant """
        return self._sense_line(self.colony.foodPheromoneGrid, steps=2, step=1.0, angle=math.pi)

    
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
        if self.ClossestFood == [-1,-1]:
            return 0
        dx = self.ClossestFood[0] - self.x
        dy = self.ClossestFood[1] - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 5:
            return 0
        # compare radians to radians
        viewAngle = math.radians(20)  # 10 degrees → radians
        angleToFood = abs(math.atan2(dy, dx) - self.direction)
        # normalize to [0, π]
        while angleToFood > math.pi:
            angleToFood -= 2*math.pi
        angleToFood = abs(angleToFood)
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
            self.turn((random.random() - 0.5) * 0.8)  # ~±0.4 rad
            return  # bail early to avoid adding the forward step
    
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
        # self.energy -= 1
        
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
        # self.energy -= 1


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
    
    def DecrementVal(self, x, y, amount=1):
        """ decrement a numeric value from the grid (for food stacking) """
        if x < 0 or x >= len(self.grid):
            return False
        if y < 0 or y >= len(self.grid[x]):
            return False
        
        current_val = self.grid[x][y]
        if current_val == [] or current_val == 0:
            return False
            
        new_val = current_val - amount
        if new_val <= 0:
            self.grid[x][y] = []
            return True
        else:
            self.grid[x][y] = new_val
            return True
    
    def IncrementVal(self, x, y, amount=1):
        """ increment a numeric value in the grid (for food stacking) """
        if x < 0 or x >= len(self.grid):
            return
        if y < 0 or y >= len(self.grid[x]):
            return
            
        current_val = self.grid[x][y]
        if current_val == []:
            current_val = 0
            
        self.grid[x][y] = current_val + amount
    
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
        # Dual pheromone system: separate grids for nest-seeking and food-seeking pheromones
        self.nestPheromoneGrid = WorldGrid(GridSize[0], GridSize[1])  # Dropped by ants NOT carrying food (paths to nest)
        self.foodPheromoneGrid = WorldGrid(GridSize[0], GridSize[1])  # Dropped by ants carrying food (paths to food)
        self.width = GridSize[0]
        self.height = GridSize[1]
        
        # Calculate field scale factor for food quantities
        # Base reference: 90x90 grid = 8100 tiles (typical for 1000x1000 screen with 11px tiles)
        self.fieldArea = self.width * self.height
        self.fieldScaleFactor = self.fieldArea / 8100.0
        
        # Scale maxFood based on field size
        self.maxFood = int(2000 * self.fieldScaleFactor)
        self.maxFood = max(100, self.maxFood)  # Minimum 100 food
        print(f'Field size: {self.width}x{self.height} = {self.fieldArea} tiles, scale factor: {self.fieldScaleFactor:.2f}, maxFood: {self.maxFood}')

        # self.hivePos = [int(GridSize[0]*0.5), int(GridSize[1]*0.5)]
        self.hivePos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
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
        
        # Stagnation detection variables
        self.lastLeaderboardChangeStep = 0  # Step when leaderboard last changed
        self.stagnationThreshold = 10000  # Steps without improvement before triggering evolution adjusters
        self.lastTopFitness = 0  # Track the best fitness achieved
        self.lastLowestLeaderboardFitness = 0  # Track the lowest fitness on the leaderboard
        
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
        
        ant.x = random.randint(0, self.width-1)
        ant.y = random.randint(0, self.height-1)
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

    def add_food(self, pos=None, amount=1):
        """add to the food grid in a random spot"""
        foodX = random.randint(0, self.width-1)
        foodY = random.randint(0, self.height-1)

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

        # Use IncrementVal to stack food instead of SetVal
        self.foodGrid.IncrementVal(foodX, foodY, amount)
 
    def add_wall(self, pos=None):
        """add to the wall grid in a random spot"""
        wallX = random.randint(0, self.width-1)
        wallY = random.randint(0, self.height-1)
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
        numWalls = int(self.width * self.height * .05)
        
            
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
        
        self.BestAnts = self.BestAnts[:100] #only keep the top 100 ants
        
        # Check if leaderboard has improved by looking at the lowest fitness
        current_lowest_fitness = 0
        if len(self.BestAnts) >= 100:  # Only check when leaderboard is full
            current_lowest_fitness = self.BestAnts[-1]["fitness"]  # Last ant has lowest fitness
            if current_lowest_fitness > self.lastLowestLeaderboardFitness:
                # Leaderboard has improved - new ants have pushed out weaker ones
                self.lastLowestLeaderboardFitness = current_lowest_fitness
                self.lastLeaderboardChangeStep = self.totalSteps
                print(f"🎯 LEADERBOARD IMPROVED! New minimum fitness: {current_lowest_fitness}")
        elif len(self.BestAnts) > 0:
            # Leaderboard not full yet, so any addition is progress
            self.lastLeaderboardChangeStep = self.totalSteps
        
        probBest = .1
        
        # print(f'Best Ants: {self.BestAnts}')
        
        #create new ants from the best ants
        whileCount = 0
        while len(self.ants) < self.maxAnts:
            whileCount += 1
            if whileCount > 1000:
                print("Stuck in while loop during repopulate, breaking out")
                break
            
            # ant = Ant()
            bestAntScore = self.BestAnts[0]["food"]
            
            
            
            #make more new ants if the best ant score is less than 10
            
            # if bestAntScore < 30:
            #     probBest = 1-(bestAntScore / 60) # the higher the score, the more likely we add a best ant
            if bestAntScore < 1:
                probBest = .2
            elif bestAntScore < 2:
                probBest = .5
            elif bestAntScore < 3:
                probBest = .75
            else:
                probBest = .99
            
            if random.random() < probBest: # if less then probBest, add a random ant
                self.add_ant(brain=None, startP=self.hivePos)
                # print("Adding random ant")
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

        if currentFood >= self.maxFood:
            return # already at max food
        
        whileCount = 0
        newFoodCount = 0
        while newFoodCount < ammt: # food scaricity produces more competition for food
            # print(f'Current Food: {currentFood} Ammt: {ammt}')
            whileCount += 1
            if whileCount > 100:
                print("Stuck in while loop during food replenish, breaking out")
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
                        newFoodCount += 1
        # print('food replenished')


    def MutateBrain(self, brain):
        """mutate the brain by changing one of the values"""
        if len(self.ants) == 0:
            return brain
        numChanges = random.randint(1, 12)
        #make sure less then half brain is changed
        numChanges = min(numChanges, int(len(brain)/2))
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

    def ApplyEvolutionAdjusters(self):
        """Apply various evolution adjusters to shake up stagnant population"""
        print("🔄 STAGNATION DETECTED! Applying evolution adjusters...")
        self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]
        # first clear all ants and set max ants to 3000
        self.ants = []
        self.maxAnts = 3000
        # self.create_world()
        
        # Adjuster 1: Introduce completely new random ants (30% of population)
        new_random_ants = int(self.maxAnts * 0.3)
        print(f"  • Adding {new_random_ants} completely new random ants")
        for i in range(new_random_ants):
            self.add_ant(brain=None, startP=self.hivePos)
        
        # Adjuster 2: Increase mutation rate for existing best ants
        if len(self.BestAnts) > 0:
            print("  • Applying heavy mutations to best ant brains")
            for ant_data in self.BestAnts[:20]:  # Mutate top 20 best ants heavily
                original_brain = ant_data["brain"].copy()
                # Apply multiple mutations
                for _ in range(3):
                    original_brain = self.MutateBrain(original_brain)
                # Add the heavily mutated ant
                new_ant = self.add_ant(brain=original_brain, startP=self.hivePos)
                new_ant.antID[1] = "EM"  # Evolution Mutated
        
        # Adjuster 3: Create hybrid brains from random combinations
        if len(self.BestAnts) >= 4:
            print("  • Creating hybrid brains from random best ant combinations")
            for i in range(10):  # Create 10 hybrid ants
                # Pick 2-4 random best ants
                num_parents = random.randint(2, min(4, len(self.BestAnts)))
                parents = random.sample(self.BestAnts, num_parents)
                
                # Create hybrid brain by randomly selecting synapses from each parent
                min_brain_size = min(len(parent["brain"]) for parent in parents)
                hybrid_brain = []
                
                for synapse_idx in range(min_brain_size):
                    # Randomly pick which parent to take this synapse from
                    chosen_parent = random.choice(parents)
                    hybrid_brain.append(chosen_parent["brain"][synapse_idx])
                
                new_ant = self.add_ant(brain=hybrid_brain, startP=self.hivePos)
                new_ant.antID[1] = "HY"  # Hybrid
        
        # # Adjuster 4: Randomize hive position to force new exploration
        # old_hive = self.hivePos.copy()
        # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]
        # print(f"  • Moved hive from {old_hive} to {self.hivePos}")
        
        # # Adjuster 5: Clear some pheromone trails to encourage new pathfinding
        # active_phers = self.pheromoneGrid.listActive()
        # clear_count = int(len(active_phers) * 0.7)  # Clear 70% of pheromones
        # phers_to_clear = random.sample(active_phers, min(clear_count, len(active_phers)))
        # for pher in phers_to_clear:
        #     self.pheromoneGrid.RemoveVal(pher[0], pher[1])
        # print(f"  • Cleared {len(phers_to_clear)} pheromone trails")
        
        # Reset stagnation counter
        self.lastLeaderboardChangeStep = self.totalSteps
        self.lastLowestLeaderboardFitness = 0  # Reset so any leaderboard improvement will be detected
        print("✅ Evolution adjusters applied! System should be less stagnant now.")
        return


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
                        
            # Proper spiral search pattern - search in expanding squares
            aX = int(ant.x)
            aY = int(ant.y)
            
            # First check the ant's current position
            current_food = self.foodGrid.GetVal(aX, aY)
            if current_food != [] and current_food > 0:
                closestFood = [aX, aY]
            else:
                # Search in expanding square rings around the ant
                for radius in range(1, 10):
                    found = False
                    
                    # Search the perimeter of the current square
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            # Only check perimeter positions (not interior)
                            if abs(dx) == radius or abs(dy) == radius:
                                checkX = aX + dx
                                checkY = aY + dy
                                
                                # Bounds check
                                if (checkX >= 0 and checkX < self.width and 
                                    checkY >= 0 and checkY < self.height):
                                    
                                    food_val = self.foodGrid.GetVal(checkX, checkY)
                                    if food_val != [] and food_val > 0:
                                        closestFood = [checkX, checkY]
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        break
                      
                      
            if closestFood != [-1,-1]:
                ant.ClossestFood = closestFood

            if self.isAtHive(ant):
                if ant.carryingFood:
                    ant.carryingFood = False
                    # Reward the ant
                    ant.life += 150
                    ant.fitness += 500 #extra reward for returning food home
                    
                    # Award navigation fitness for successful return trip
                    if ant.foodPickupPos is not None:
                        nav_fitness = ant.calculateNavigationFitness([ant.x, ant.y])
                        ant.navigationFitness = max(0, nav_fitness)  # Only positive navigation fitness for successful return
                        ant.fitness += ant.navigationFitness
                        # print(f'Ant {ant.antID} navigation fitness: {ant.navigationFitness}')
                        ant.foodPickupPos = None  # Reset pickup position
                    
                    # print(f'Ant {ant.antID} returned food to the hive!!')
                
            ant.RunBrain()
            ant.pDirection = float(ant.direction)
           
           
           
           # history saving
            lastknownPos = ant.posHistory[-1] if len(ant.posHistory) > 0 else [0,0]
     
            moveDist = math.sqrt((ant.x - lastknownPos[0])**2 + (ant.y - lastknownPos[1])**2)
        # limited to 1 tile per move to prevent too much history
            if abs(moveDist) > 1:
                ant.posHistory.append([ant.x, ant.y])
           ## end history saving

            ant.life -= 1

            
            #check if the ant has a closestFood value, if so detect the distance and if close enough, consume the food
            
            antClosestFood = ant.ClossestFood
            if antClosestFood != [-1,-1]:
                distToFood = math.sqrt((ant.x - antClosestFood[0])**2 + (ant.y - antClosestFood[1])**2)
                if distToFood < 1: #give a little leeway
                    if ant.carryingFood == False: #if the ant is not carrying food force ants to return home to keep eating

                        # Use DecrementVal instead of RemoveVal to support food stacking
                        if self.foodGrid.DecrementVal(antClosestFood[0], antClosestFood[1]):
                        # print(f'consuming 1 food at {antClosestFood}')
                            ant.ClossestFood = [-1,-1]
                            # ant.energy += 10
                            ant.FoodConsumed += 1
                            ant.life += 150 #reward the ant for finding food
                            ant.fitness += 1
                            if ant.life > 400:
                                ant.life = 400 #keep things reasonable, some ants are too good

                            ant.carryingFood = True
                            # Record the pickup location for navigation fitness calculation
                            ant.foodPickupPos = [int(ant.x), int(ant.y)]
                            ant.navigationFitness = 0  # Reset navigation fitness

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
            # Pheromone dropping only when ant moves to a new cell
            x_pos = int(ant.x)
            y_pos = int(ant.y)
            
            # Check if ant has moved to a new cell
            if ant.prevCellX != x_pos or ant.prevCellY != y_pos:
                pheromone_amount = 1.0  # Higher amount since it's dropped less frequently
                
                if ant.carryingFood:
                    # Ant carrying food drops food pheromone (path to food)
                    getCurrentPher = self.foodPheromoneGrid.GetVal(x_pos, y_pos)
                    if getCurrentPher == [] or getCurrentPher == None:
                        getCurrentPher = 0
                    newAmmt = getCurrentPher + pheromone_amount
                    if newAmmt < 10:
                        self.foodPheromoneGrid.SetVal(x_pos, y_pos, newAmmt)
                else:
                    # Ant not carrying food drops nest pheromone (path to nest)
                    getCurrentPher = self.nestPheromoneGrid.GetVal(x_pos, y_pos)
                    if getCurrentPher == [] or getCurrentPher == None:
                        getCurrentPher = 0
                    newAmmt = getCurrentPher + pheromone_amount
                    if newAmmt < 10:
                        self.nestPheromoneGrid.SetVal(x_pos, y_pos, newAmmt)
                
                # Update previous cell position
                ant.prevCellX = x_pos
                ant.prevCellY = y_pos
                
            # #check if front pos has pheromones (both types)
            # nestPherVal = self.nestPheromoneGrid.GetVal(int(frontPos[0]), int(frontPos[1]))
            # foodPherVal = self.foodPheromoneGrid.GetVal(int(frontPos[0]), int(frontPos[1]))
            
            # # Update ant's pheromone sensors
            # if nestPherVal != [] and nestPherVal != None and nestPherVal > 0:
            #     ant.nestPherFront = nestPherVal
            # else:
            #     ant.nestPherFront = 0
                
            # if foodPherVal != [] and foodPherVal != None and foodPherVal > 0:
            #     ant.foodPherFront = foodPherVal
            # else:
            #     ant.foodPherFront = 0
                

                
            
            # for food in self.food:
            #     if ant.x == food[0] and ant.y == food[1]:
            #         ant.energy += 10
            #         self.food.remove(food)
            # for wall in self.walls:
            #     if ant.x == wall[0] and ant.y == wall[1]:
            #         ant.energy -= 10

        # print('ants updated')
        
        #remove dead ants
        for ant in self.ants[:]:
            if ant.life <= 0: #DEAD ANT
                #if debugbrain, pick another ant
                if ant.DebugBrain:
                    self.FollowNextAnt = True
                
                # Calculate navigation fitness if ant was carrying food when it died
                if ant.carryingFood and ant.foodPickupPos is not None:
                    ant.navigationFitness = ant.calculateNavigationFitness()
                    ant.fitness += ant.navigationFitness  # Can be positive or negative
                
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
                    # Update top fitness for tracking purposes (not stagnation)
                    if antFitness > self.lastTopFitness:
                        self.lastTopFitness = antFitness
            # topFoodCount = self.BestAnts[0]["food"] if len(self.BestAnts) > 0 else 0
        
        
        #remove old pheromone cells and decay both types

        # Handle nest pheromones
        activeNestPhers = self.nestPheromoneGrid.listActive()
        for pher in activeNestPhers:
            if pher[2] <= 0:
                self.nestPheromoneGrid.RemoveVal(pher[0], pher[1])
            else:
                #decay the pheromone
                self.nestPheromoneGrid.SetVal(pher[0], pher[1], pher[2]-.005)
        
        # Handle food pheromones  
        activeFoodPhers = self.foodPheromoneGrid.listActive()
        for pher in activeFoodPhers:
            if pher[2] <= 0:
                self.foodPheromoneGrid.RemoveVal(pher[0], pher[1])
            else:
                #decay the pheromone
                self.foodPheromoneGrid.SetVal(pher[0], pher[1], pher[2]-.005)

        # print('pheromone updated')
        quadrant = int(self.totalSteps / 200) % 4
        # Scale food replenishment amounts based on field size
        baseSmall = max(1, int(100 * self.fieldScaleFactor))
        baseLarge = max(10, int(800 * self.fieldScaleFactor))
        baseCluster = max(1, int(25 * self.fieldScaleFactor))
        
        # Add more food clusters in all quadrants for better distribution
        self.ReplenishFood((quadrant+1)% 4, baseSmall)
        self.ReplenishFood((quadrant+2)% 4, baseSmall)
        self.ReplenishFood((quadrant+3)% 4, baseSmall)
        self.ReplenishFood(quadrant, baseLarge)
        
        # Add additional smaller clusters in all quadrants every step
        for q in range(4):
            self.ReplenishFood(q, baseCluster)
        # print('food replenished')

        repop_result = self.Repopulate()
        endTime = time.time()
        self.UpdateTime = endTime - startTime
        
        # Check for stagnation and apply evolution adjusters if needed
        steps_since_improvement = self.totalSteps - self.lastLeaderboardChangeStep
        if steps_since_improvement >= self.stagnationThreshold:
            self.ApplyEvolutionAdjusters()
        
        if self.totalSteps % 1000 == 0:
            print("------------------report-----------------")
            print(f'Steps So Far: {self.totalSteps}')
            print(f'Total Ants: {len(self.ants)}')
            print(f'Total Dead Ants: {self.totalDeadAnts}')
            print(f'Top Food Found: {self.topFoodFound}')
            print(f'Best Fitness: {self.lastTopFitness}')
            print(f'Steps Since Leaderboard Change: {steps_since_improvement}/{self.stagnationThreshold}')
            print(f'Current Leaderboard Size: {len(self.BestAnts)}')
            if len(self.BestAnts) >= 100:
                print(f'Lowest Leaderboard Fitness: {self.lastLowestLeaderboardFitness}')
            # print(f'Top Pheromone: {self.TopPheromone}')
            print(f'Update Time: {self.UpdateTime}')
            if "probbest" in repop_result:
                print(f'Repopulate Probiability of random ant: {repop_result["probbest"]}')
            print("----------------------------------------")
            #randomize the hive pos
            # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]


        # if self.totalSteps % 10000 == 0:
            # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]

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
        
        # print("Drawing Paths")

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
                    # print(f'Ant {ant.antID} has posHistory length: {len(ant.posHistory)}')
                    #ant must have eaten one food
                    if ant.FoodConsumed > 0:
                        # print(f'Ant {ant.antID} drawing path, food consumed: {ant.FoodConsumed}')
                        for i in range(len(ant.posHistory) - 1):
                            p1 = self.WorldToScreen(ant.posHistory[i])
                            p2 = self.WorldToScreen(ant.posHistory[i + 1])
                            pygame.draw.line(screen, ant.Color, p1, p2, 1)
                        
        timeDelta = time.time() - self.LastSave
        timeDistance = 60 #save every minute
      


        if timeDelta > timeDistance:
            #save the best ants to a file
            self.saveData()
            #also save an image of the screen (only if not Pi Mode)
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

        # Find maximum pheromone values for both types for proper scaling
        maxNestPher = 0
        maxFoodPher = 0
        
        for pher in self.nestPheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal > maxNestPher:
                maxNestPher = pherVal
                
        for pher in self.foodPheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal > maxFoodPher:
                maxFoodPher = pherVal
            
        self.TopPheromone = max(maxNestPher, maxFoodPher)
        
        # Draw nest pheromones (blue - paths to nest)
        for pher in self.nestPheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal <= 0:
                continue
            ppxy = self.WorldToScreen([pher[0], pher[1]])
            
            if maxNestPher > 0:
                colorVal = pherVal / maxNestPher * 255.0
                if colorVal < 10:
                    colorVal = 10
                
                ppxy = (int(ppxy[0]), int(ppxy[1]))
                pygame.draw.rect(screen, (0, 0, colorVal), ((ppxy[0]), (ppxy[1]), int(self.TileSize), int(self.TileSize)))
        
        # Draw food pheromones (red - paths to food)
        for pher in self.foodPheromoneGrid.listActive():
            pherVal = pher[2]
            if pherVal <= 0:
                continue
            ppxy = self.WorldToScreen([pher[0], pher[1]])
            
            if maxFoodPher > 0:
                colorVal = pherVal / maxFoodPher * 255.0
                if colorVal < 10:
                    colorVal = 10
                
                ppxy = (int(ppxy[0]), int(ppxy[1]))
                pygame.draw.rect(screen, (colorVal, 0, 0), ((ppxy[0]), (ppxy[1]), int(self.TileSize), int(self.TileSize)))
        timeDelta = time.time() - self.LastSave
        timeDistance = 60 #save every minute
        if isPi:
            timeDistance = 60*60 # hour #save every hour for the pi

        if timeDelta > timeDistance:
            #save the best ants to a file
            self.saveData()
            #also save an image of the screen (only if not Pi Mode)
            if isPi == False:
                tCode = time.strftime("%Y%m%d-%H%M%S")
                pygame.image.save(screen, f'dataSave/{tCode}.png')
            self.LastSave = time.time()

        for food in self.foodGrid.listActive():
            food_count = food[2]
            if food_count == 0:
                continue
            fpxy = self.WorldToScreen(food)
            fpxy = (int(fpxy[0]), int(fpxy[1]))
            
            # Calculate visual intensity based on food stack count
            # Cap at 10 for visual purposes to prevent overly bright colors
            visual_count = min(food_count, 10)
            
            # Base green color that gets brighter with more food
            base_green = 100
            intensity_green = base_green + (visual_count * 15)  # Gets brighter with more food
            intensity_green = min(intensity_green, 255)
            
            cx = fpxy[0] + self.TileSize/2
            cy = fpxy[1] + self.TileSize/2
            
            # Draw multiple concentric circles to show "stacking" effect
            if food_count == 1:
                # Single food - simple circle
                pygame.draw.circle(screen, (0, intensity_green, 0), (int(cx), int(cy)), int(self.TileSize/2), 1)
            else:
                # Multiple food - show stacking with filled circle and rings
                # Fill the center for stacked food
                pygame.draw.circle(screen, (0, intensity_green//2, 0), (int(cx), int(cy)), int(self.TileSize/3))
                
                # Draw concentric rings to show stacking depth
                for ring in range(min(3, visual_count)):  # Max 3 rings for visual clarity
                    ring_radius = int(self.TileSize/2) - ring * 2
                    if ring_radius > 0:
                        ring_intensity = intensity_green - (ring * 30)
                        ring_intensity = max(ring_intensity, 50)  # Keep minimum visibility
                        pygame.draw.circle(screen, (0, ring_intensity, 0), (int(cx), int(cy)), ring_radius, 1)
                
                # Add a small text indicator for high food counts (only if not Pi mode)
                if food_count > 5 and not isPi:
                    font = pygame.font.Font(None, 12)
                    text = font.render(str(food_count), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(int(cx), int(cy)))
                    screen.blit(text, text_rect)
            
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
                #pheromone sensors
                # text = font.render(f'NEST PHER F:{ant.nestPherFront}', True, (255,255,255))
                # screen.blit(text, (pxy[0]+10, pxy[1]+60))
                # text = font.render(f'FOOD PHER F:{ant.foodPherFront}', True, (255,255,255))
                # screen.blit(text, (pxy[0]+10, pxy[1]+75))
                #close to food
                text = font.render(f'CLOSE To FOOD:{ant.closeToFood()}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+90))
                #foodDir
                text = font.render(f'FOOD DIR:{ant.foodDir()}', True, (255,255,255))
                screen.blit(text, (pxy[0]+10, pxy[1]+105))
                
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
                        
                    
                
                   

        # Draw the hive position with a cool spinning shape (draw last to avoid overlap)
        self.drawHive(screen, isPi)

        # Check mouse position to determine if stats should be shown
        mouse_pos = pygame.mouse.get_pos()
        show_stats = mouse_pos[1] <= 20  # Show stats if mouse is within 20px of top

        if show_stats:
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
                ColorLookup = {"CL":(200, 0, 200), "M":(255, 200, 0), "CH":(0, 200, 0), "N":(255, 0, 0), "EM":(255, 100, 255), "HY":(100, 255, 255)}
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
        
    def drawHive(self, screen, isPi=False):
        """Draw a cool spinning hive visualization"""
        # Convert hive position to screen coordinates
        hive_screen_pos = self.WorldToScreen(self.hivePos)
        hive_x = int(hive_screen_pos[0])
        hive_y = int(hive_screen_pos[1])
        
        if isPi:
            # Simple circle for Pi mode - lightweight rendering
            pygame.draw.circle(screen, (255, 215, 0), (hive_x, hive_y), 15, 3)
            pygame.draw.circle(screen, (255, 255, 0), (hive_x, hive_y), 8, 2)
            return
        
        # Get current time for animation
        current_time = time.time()
        
        # Create spinning effect - multiple rotating shapes
        rotation_speed = 2.0  # radians per second
        base_angle = current_time * rotation_speed
        
        # Draw multiple rotating hexagons/shapes of different sizes
        hive_colors = [(255, 215, 0), (255, 165, 0), (255, 140, 0)]  # Gold to orange gradient
        
        for layer in range(3):
            angle_offset = base_angle + (layer * math.pi / 3)  # Each layer rotates at different phase
            radius = 15 + (layer * 8)  # Different sizes
            color = hive_colors[layer]
            
            # Draw hexagon
            points = []
            for i in range(6):
                point_angle = angle_offset + (i * math.pi / 3)
                point_x = hive_x + math.cos(point_angle) * radius
                point_y = hive_y + math.sin(point_angle) * radius
                points.append((int(point_x), int(point_y)))
            
            # Draw the hexagon outline
            if len(points) >= 3:
                pygame.draw.polygon(screen, color, points, 2)
        
        # Draw central core - pulsating circle
        pulse_radius = 8 + int(3 * math.sin(current_time * 4))  # Pulsing effect
        pygame.draw.circle(screen, (255, 255, 0), (hive_x, hive_y), pulse_radius, 3)
        
        # Add some sparkle effects around the hive
        for i in range(8):
            sparkle_angle = base_angle * 0.5 + (i * math.pi / 4)
            sparkle_radius = 35 + int(5 * math.sin(current_time * 3 + i))
            sparkle_x = hive_x + math.cos(sparkle_angle) * sparkle_radius
            sparkle_y = hive_y + math.sin(sparkle_angle) * sparkle_radius
            
            # Draw small sparkles
            sparkle_size = 2 + int(math.sin(current_time * 6 + i))
            if sparkle_size > 0:
                pygame.draw.circle(screen, (255, 255, 255), (int(sparkle_x), int(sparkle_y)), sparkle_size)

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
        
        # Pixel scaling for Pi mode - each logical pixel becomes NxN screen pixels
        # Set to 1 for normal resolution, 2 for half-res (4 pixels per logical pixel), etc.
        self.pixelScale = 1  # Default: no scaling
        
        self.screenSize = (1000, 1000)
        self.renderSize = (1000, 1000)  # Actual rendering resolution (may be smaller for scaled modes)
        #using argparse
        parser = argparse.ArgumentParser(description='Run the ant simulation')
        parser.add_argument('--pi', action='store_true', help='Run on a Raspberry Pi')
        parser.add_argument('--paths', action='store_true', help='Draw the paths of the ants')
        #add debug mode
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        parser.add_argument('--scale', type=int, default=2, help='Pixel scale factor for Pi mode (1=full res, 2=half res, etc.)')
        args = parser.parse_args()
        if args.pi:
            self.isPi = True
            print('Running on Raspberry Pi')
            self.screenSize = (480, 1920)
            # Apply pixel scaling for Pi mode
            self.pixelScale = max(1, args.scale)  # Minimum scale of 1
            print(f'Pixel scale: {self.pixelScale}x (each logical pixel = {self.pixelScale}x{self.pixelScale} screen pixels)')

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
        
        tileSize = 11  #smaller is slower
                
        if self.isPi:
            print('Starting display on PI')
            pygame.display.init()
            pygame.display.list_modes()
            os.environ["SDL_VIDEODRIVER"] = "rbcon" # or maybe 'fbcon'
            #set display environment variables
            os.environ["DISPLAY"] = ":0.0"
            # Set up display at full screen resolution
            self.screen = pygame.display.set_mode(self.screenSize, pygame.FULLSCREEN)
            
            # Create a smaller render surface for scaled mode
            # Each logical pixel will be pixelScale x pixelScale screen pixels
            self.renderSize = (self.screenSize[0] // self.pixelScale, self.screenSize[1] // self.pixelScale)
            self.renderSurface = pygame.Surface(self.renderSize)
            print(f'Render size: {self.renderSize} -> scaled to {self.screenSize}')
            
            # Initialize Pygame
            print('Hiding cursor')
            pygame.mouse.set_visible(False) # Hide cursor here

            self.maxAnts = 80
            # Adjust tile size for the scaled render resolution
            # Smaller render surface needs proportionally larger tiles to maintain same grid density
            tileSize = 25 // self.pixelScale
            if tileSize < 5:
                tileSize = 5  # Minimum tile size
            print(f'Tile size adjusted to: {tileSize}')
        
        else:
            # os.environ["SDL_VIDEO_WINDOW_POS"] = "-1100,0"
            self.screen = pygame.display.set_mode(self.screenSize)
            self.renderSize = self.screenSize
            self.renderSurface = self.screen  # No scaling needed, render directly to screen
            
            
        self.clock = pygame.time.Clock()

           
        print('Creating Ant Colony')
        # Use renderSize for the ant colony so the grid matches the render resolution
        self.antColony = AntColony(self.renderSize, self.maxAnts, tileSize)
        
        # self.antColony.LoadBestAnts( self.maxAnts )

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
                self.antColony.drawPaths(self.renderSurface, isPi= self.isPi)
            else:
                self.antColony.drawAnts(self.renderSurface, isPi=self.isPi)
                text = f'FPS: {fps}'
                font = pygame.font.Font(None, 26)
                text = font.render(text, True, (255, 255, 255))
                self.renderSurface.blit(text, (self.renderSize[0]-100, 10))
                #num ants displayed
                text = f'Ants: {len(self.antColony.ants)}'
                font = pygame.font.Font(None, 26)
                text = font.render(text, True, (255, 255, 255))
                self.renderSurface.blit(text, (self.renderSize[0]-100, 30))
            
            # Scale up the render surface to the screen if using pixel scaling
            if self.isPi and self.pixelScale > 1:
                # Use NEAREST neighbor scaling to maintain sharp pixels (no blurring)
                scaled_surface = pygame.transform.scale(self.renderSurface, self.screenSize)
                self.screen.blit(scaled_surface, (0, 0))
            elif self.renderSurface is not self.screen:
                # Non-Pi mode with separate render surface (shouldn't happen normally)
                self.screen.blit(self.renderSurface, (0, 0))
            
            # Draw battery indicator directly on screen (not affected by scaling)
            # Minimal 20px bar on bottom edge, no text
            if self.isPi and PISUGAR_AVAILABLE:
                battery_bar_height = 20
                battery_bar_y = self.screenSize[1] - battery_bar_height
                battery_bar_width = self.screenSize[0]
                battery_level = self.antColony.batteryLevel
                
                if battery_level > 0:
                    battery_width = int((battery_level / 100.0) * battery_bar_width)
                    
                    # Color based on battery level
                    if battery_level > 50:
                        battery_color = (0, 255, 0)  # Green
                    elif battery_level > 20:
                        battery_color = (255, 255, 0)  # Yellow
                    else:
                        battery_color = (255, 0, 0)  # Red
                    
                    pygame.draw.rect(self.screen, battery_color, (0, battery_bar_y, battery_width, battery_bar_height))

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
            
            # Adaptive ant count based on FPS
            if self.isPi:
                # Pi mode: target lower FPS, be more aggressive with scaling
                if fps < 3:
                    self.maxAnts -= 5
                    if self.maxAnts < 5:
                        self.maxAnts = 5
                elif fps > 8:
                    self.maxAnts += 5
                    if self.maxAnts > 200:
                        self.maxAnts = 200
                self.antColony.maxAnts = self.maxAnts
            else:
                # Desktop mode
                if fps < 2:
                    self.maxAnts -= 1
                    print(f'Reducing max ants to {self.maxAnts} for performance')
                    if self.maxAnts < 100:
                        self.maxAnts = 100
                    self.antColony.maxAnts = self.maxAnts
                elif fps > 10:
                    self.maxAnts += 1
                    self.antColony.maxAnts = self.maxAnts

            pygame.display.flip()
            # self.clock.tick(120)
            self.clock.tick()

        pygame.quit()

if __name__ == "__main__":
    print('Starting Simulation')
    game = Game()
    game.run()
