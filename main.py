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

# Brain size limits (global constants)
MIN_BRAIN_SIZE = 6
MAX_BRAIN_SIZE = 64

# Pheromone limits
MAX_PHEROMONE = 10  # Maximum pheromone value per cell


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
    # Pre-computed trig values for 4 cardinal directions (front, left, right, back)
    # Keys are angle offsets, values are (cos, sin) tuples
    _TRIG_CACHE = {
        0.0: (1.0, 0.0),                    # Front
        -math.pi/2: (0.0, -1.0),            # Left
        math.pi/2: (0.0, 1.0),              # Right  
        math.pi: (-1.0, 0.0),               # Back
    }
    
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
            self.isCarryingFood,
            # Terrain density sensors (sense how hard terrain is in each direction)
            self.getTerrainFront, self.getTerrainLeft, 
            self.getTerrainRight, self.getTerrainBack
        ]

       
        self.OutputDestinations = [ self.move, self.turn ]

        # # Dual pheromone tracking
        # self.nestPherFront = 0
        # self.foodPherFront = 0
        
        self.posHistory = []
        
        self.DebugBrain = False
        
        self.Color = []
        
        # Cached sensor values - computed once per frame for performance
        self._cached_nest_pher_front = 0
        self._cached_nest_pher_left = 0
        self._cached_nest_pher_right = 0
        self._cached_nest_pher_back = 0
        self._cached_food_pher_front = 0
        self._cached_food_pher_left = 0
        self._cached_food_pher_right = 0
        self._cached_food_pher_back = 0
        self._cached_hive_direction = 0
        self._cached_hive_distance = 0
        # Cached terrain density sensors (normalized 0-1)
        self._cached_terrain_front = 0
        self._cached_terrain_left = 0
        self._cached_terrain_right = 0
        self._cached_terrain_back = 0
        self._sensors_computed = False  # Flag to track if sensors need recomputing
        


    def create_brain(self, size=None):
        """Create_Brain : create a random brain"""
        if size is None:
            size = MAX_BRAIN_SIZE
        brainSize = random.randint(MIN_BRAIN_SIZE, size)
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
    
    def precompute_sensors(self):
        """Precompute all sensor values once per frame for performance.
        This avoids redundant grid lookups and trig calculations during RunBrain()."""
        
        # Cache direction trig values once
        cos_dir = math.cos(self.direction)
        sin_dir = math.sin(self.direction)
        
        # Precompute all 8 pheromone sensors using inline calculations
        # All pheromone values normalized to 0-1 range by dividing by MAX_PHEROMONE
        # Front (angle=0): dx=cos_dir, dy=sin_dir
        self._cached_nest_pher_front = self._sense_line_fast(
            self.colony.nestPheromoneGrid, cos_dir, sin_dir, steps=3) / MAX_PHEROMONE
        self._cached_food_pher_front = self._sense_line_fast(
            self.colony.foodPheromoneGrid, cos_dir, sin_dir, steps=3) / MAX_PHEROMONE
        
        # Left (angle=-pi/2): rotate by -90 degrees
        # dx = cos_dir*0 - sin_dir*(-1) = sin_dir
        # dy = cos_dir*(-1) + sin_dir*0 = -cos_dir
        dx_left, dy_left = sin_dir, -cos_dir
        self._cached_nest_pher_left = self._sense_line_fast(
            self.colony.nestPheromoneGrid, dx_left, dy_left, steps=3) / MAX_PHEROMONE
        self._cached_food_pher_left = self._sense_line_fast(
            self.colony.foodPheromoneGrid, dx_left, dy_left, steps=3) / MAX_PHEROMONE
        
        # Right (angle=pi/2): rotate by +90 degrees  
        # dx = cos_dir*0 - sin_dir*1 = -sin_dir
        # dy = cos_dir*1 + sin_dir*0 = cos_dir
        dx_right, dy_right = -sin_dir, cos_dir
        self._cached_nest_pher_right = self._sense_line_fast(
            self.colony.nestPheromoneGrid, dx_right, dy_right, steps=3) / MAX_PHEROMONE
        self._cached_food_pher_right = self._sense_line_fast(
            self.colony.foodPheromoneGrid, dx_right, dy_right, steps=3) / MAX_PHEROMONE
        
        # Back (angle=pi): rotate by 180 degrees
        # dx = -cos_dir, dy = -sin_dir
        dx_back, dy_back = -cos_dir, -sin_dir
        self._cached_nest_pher_back = self._sense_line_fast(
            self.colony.nestPheromoneGrid, dx_back, dy_back, steps=2) / MAX_PHEROMONE
        self._cached_food_pher_back = self._sense_line_fast(
            self.colony.foodPheromoneGrid, dx_back, dy_back, steps=2) / MAX_PHEROMONE
        
        # Cache hive direction and distance (also computed multiple times)
        dx_hive = self.colony.hivePos[0] - self.x
        dy_hive = self.colony.hivePos[1] - self.y
        
        # Hive direction
        angle_to_hive = math.atan2(dy_hive, dx_hive)
        relative_angle = angle_to_hive - self.direction
        # Normalize to [-pi, pi] using modulo (faster than while loops)
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        self._cached_hive_direction = relative_angle / math.pi
        
        # Hive distance
        distance = math.sqrt(dx_hive * dx_hive + dy_hive * dy_hive)
        max_distance = math.sqrt(self.colony.width**2 + self.colony.height**2)
        self._cached_hive_distance = 1 - (distance / max_distance)
        
        # Precompute terrain density sensors (normalized 0-1, where 1 = max density)
        # Front
        self._cached_terrain_front = self._sense_terrain_fast(cos_dir, sin_dir)
        # Left
        self._cached_terrain_left = self._sense_terrain_fast(dx_left, dy_left)
        # Right
        self._cached_terrain_right = self._sense_terrain_fast(dx_right, dy_right)
        # Back
        self._cached_terrain_back = self._sense_terrain_fast(dx_back, dy_back)
        
        self._sensors_computed = True
    
    def _sense_terrain_fast(self, dx, dy, steps=2):
        """Sense terrain density in a direction, returns normalized value 0-1
        Walls (WALL_DENSITY = -1) are treated as maximum density (impassable)"""
        total_density = 0.0
        x, y = self.x, self.y
        max_density = self.colony.maxTerrainDensity
        for i in range(1, steps + 1):
            ax = int(x + dx * i)
            ay = int(y + dy * i)
            val = self.colony.terrainGrid.GetVal(ax, ay)
            if val not in ([], False, None):
                # Treat walls as maximum density so ants sense them as impassable
                if val == self.colony.WALL_DENSITY:
                    total_density += max_density
                else:
                    total_density += val
        # Normalize: max possible is steps * maxTerrainDensity
        # Return average density normalized to 0-1
        return (total_density / steps) / max_density
    
    def _sense_line_fast(self, grid, dx, dy, steps=3, step=1.0):
        """Fast sensor line sensing with pre-computed direction vectors"""
        acc = 0.0
        x, y = self.x, self.y
        for i in range(1, steps + 1):
            ax = int(x + dx * (i * step))
            ay = int(y + dy * (i * step))
            v = grid.GetVal(ax, ay)
            if v not in ([], False, None):
                acc += v
        return acc / steps
           
    def calculateNavigationFitness(self, currentPos=None):
        """
        Calculate fitness based on actual PROGRESS toward hive.
        Returns positive value for progress toward hive, negative for moving away.
        This rewards efficient navigation, not circuitous paths.
        """
        if self.foodPickupPos is None:
            return 0
        
        if currentPos is None:
            currentPos = [self.x, self.y]
        
        # Distance from where food was picked up to the hive
        pickup_to_hive = math.sqrt((self.foodPickupPos[0] - self.colony.hivePos[0])**2 + 
                                   (self.foodPickupPos[1] - self.colony.hivePos[1])**2)
        
        # Distance from current position to the hive
        current_to_hive = math.sqrt((currentPos[0] - self.colony.hivePos[0])**2 + 
                                    (currentPos[1] - self.colony.hivePos[1])**2)
        
        # Progress = how much CLOSER to hive the ant got (positive = moved toward hive)
        progress = pickup_to_hive - current_to_hive
        
        # 5 points per tile of actual progress toward hive
        # Cap penalty to prevent catastrophic fitness loss for wandering
        if progress >= 0:
            return int(progress * 5)
        else:
            return max(-100, int(progress * 5))  # Cap penalty at -100
    
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
        
        # Use cached trig for cardinal directions, compute once for others
        if angle in Ant._TRIG_CACHE:
            # Rotate cached unit vector by ant's direction
            cached_cos, cached_sin = Ant._TRIG_CACHE[angle]
            dir_cos = math.cos(self.direction)
            dir_sin = math.sin(self.direction)
            # Rotation formula: (x', y') = (x*cos - y*sin, x*sin + y*cos)
            dx = cached_cos * dir_cos - cached_sin * dir_sin
            dy = cached_cos * dir_sin + cached_sin * dir_cos
        else:
            # Fallback for non-cardinal angles
            dx = math.cos(self.direction + angle)
            dy = math.sin(self.direction + angle)
        
        for i in range(1, steps+1):
            ax = int(self.x + dx * (i*step))
            ay = int(self.y + dy * (i*step))
            v = grid.GetVal(ax, ay)
            if v not in ([], False, None):
                acc += v
        return acc / steps

    # Nest pheromone sensors (dropped by ants NOT carrying food - paths to nest)
    # These now return cached values computed by precompute_sensors()
    def getNestPherFront(self):
        """getNestPherFront() : returns the amount of nest pheromone in front of the ant """
        return self._cached_nest_pher_front
    
    def getNestPherLeft(self):
        """getNestPherLeft() : returns the amount of nest pheromone to the left of the ant """
        return self._cached_nest_pher_left

    def getNestPherRight(self):
        """getNestPherRight() : returns the amount of nest pheromone to the right of the ant """
        return self._cached_nest_pher_right

    def getNestPherBack(self):
        """getNestPherBack() : returns the amount of nest pheromone behind the ant """
        return self._cached_nest_pher_back

    # Food pheromone sensors (dropped by ants carrying food - paths to food)
    def getFoodPherFront(self):
        """getFoodPherFront() : returns the amount of food pheromone in front of the ant """
        return self._cached_food_pher_front
    
    def getFoodPherLeft(self):
        """getFoodPherLeft() : returns the amount of food pheromone to the left of the ant """
        return self._cached_food_pher_left

    def getFoodPherRight(self):
        """getFoodPherRight() : returns the amount of food pheromone to the right of the ant """
        return self._cached_food_pher_right

    def getFoodPherBack(self):
        """getFoodPherBack() : returns the amount of food pheromone behind the ant """
        return self._cached_food_pher_back

    
    def getHiveDirection(self):
        """getHiveDirection() : returns the relative direction to the hive """
        return self._cached_hive_direction

    def getHiveDistance(self):
        """getHiveDistance() : returns the normalized inverse distance to the hive """
        return self._cached_hive_distance


    def isCarryingFood(self):
        """isCarryingFood() : returns 1 if the ant is carrying food, else 0 """
        return 1 if self.carryingFood else 0

    # Terrain density sensors - return normalized values (0-1) where 1 = max density
    def getTerrainFront(self):
        """getTerrainFront() : returns terrain density in front (0-1, higher = harder) """
        return self._cached_terrain_front
    
    def getTerrainLeft(self):
        """getTerrainLeft() : returns terrain density to the left (0-1, higher = harder) """
        return self._cached_terrain_left
    
    def getTerrainRight(self):
        """getTerrainRight() : returns terrain density to the right (0-1, higher = harder) """
        return self._cached_terrain_right
    
    def getTerrainBack(self):
        """getTerrainBack() : returns terrain density behind (0-1, higher = harder) """
        return self._cached_terrain_back

    def foodDir(self):
        """foodDir() : returns the direction of the closest food """
        # return the relative direction of the closest food in a value between -1 and 1
        # if there is no food return 0
        if self.ClossestFood == [-1,-1]:
            return 0
        
        #get the direction of the food
        dx = self.ClossestFood[0] - self.x
        dy = self.ClossestFood[1] - self.y
        
        angle = math.atan2(dy, dx)
        relative_angle = angle - self.direction
        # Normalize to [-pi, pi] to ensure output is [-1, 1]
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        return relative_angle / math.pi
 
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
    
    def move(self,amount):
        """move() : move the ant in the direction it is facing """
        #limit amount to 5
        if amount > .8:
            amount = .8
        if amount < -.8:
            amount = -.8
        
        
        amount*=.5
       
       # ant has no access to the grid, so it cannot check for walls
        # futurePox = [self.x + math.cos(self.direction) * amount, self.y + math.sin(self.direction) * amount]
        # curWall = self.terrainGrid.GetVal(int(futurePox[0]), int(futurePox[1])
        # if curWall != False:
        #     amount = 0
        #     self.life -= 1# penalize the ant for hitting a wall
        #     #damage the wall at the front of the ant by .1
        #     self.terrainGrid.SetVal(int(futurePox[0]), int(futurePox[1]), curWall - .1)
        
        if self.DebugBrain:
            print(self.blockedFront)
                  
        if self.blockedFront!= False:
            amount = 0
            self.life -= 1# penalize the ant for hitting a wall
            #damage the wall at the front of the ant by .1
            self.turn((random.random() - 0.5) * 0.8)  # ~±0.4 rad
            return  # bail early to avoid adding the forward step
    
        if self.DebugBrain:
            print(f'moving amount: {amount}')
    
        addX = math.cos(self.direction) * amount
        addY = math.sin(self.direction) * amount
        if self.DebugBrain:
            print(f'addX: {addX}, addY: {addY}')
        
        self.x += addX
        self.y += addY
        
        # Clamp position to grid boundaries - prevent ants from escaping
        if self.x < 0:
            self.x = 0
        elif self.x >= self.colony.width:
            self.x = self.colony.width - 0.01
        if self.y < 0:
            self.y = 0
        elif self.y >= self.colony.height:
            self.y = self.colony.height - 0.01
        
        if self.DebugBrain:
            print(f'new pos: {self.x}, {self.y}')
        # self.energy -= 1
        
        # set farthest traveled if greater than current
        self.FarthestTraveled = max(self.FarthestTraveled, math.sqrt((self.x - self.colony.hivePos[0])**2 + (self.y - self.colony.hivePos[1])**2))
    
    def turn(self, direction):
        """turn() : turn the ant in a direction """
        self.direction += direction
        # Normalize direction to [0, 2π) using modulo
        self.direction = self.direction % (2 * math.pi)


class SpatialFoodIndex:
    """Spatial hash for O(1) nearest food lookup instead of O(r²) spiral search"""
    
    def __init__(self, width, height, bucket_size=10):
        self.bucket_size = bucket_size
        self.width = width
        self.height = height
        self.buckets_x = (width + bucket_size - 1) // bucket_size
        self.buckets_y = (height + bucket_size - 1) // bucket_size
        # Each bucket is a set of (x, y) food positions
        self.buckets = [[set() for _ in range(self.buckets_y)] for _ in range(self.buckets_x)]
    
    def _get_bucket(self, x, y):
        """Get bucket indices for a world position"""
        bx = max(0, min(x // self.bucket_size, self.buckets_x - 1))
        by = max(0, min(y // self.bucket_size, self.buckets_y - 1))
        return bx, by
    
    def add(self, x, y):
        """Add a food position to the spatial index"""
        bx, by = self._get_bucket(x, y)
        self.buckets[bx][by].add((x, y))
    
    def remove(self, x, y):
        """Remove a food position from the spatial index"""
        bx, by = self._get_bucket(x, y)
        self.buckets[bx][by].discard((x, y))
    
    def find_nearest(self, x, y, max_radius):
        """
        Find nearest food within max_radius using Chebyshev distance (ring-based).
        Returns [fx, fy] or [-1, -1] if no food found.
        
        Uses Chebyshev distance (max(|dx|,|dy|)) to match the original spiral
        search behavior where rings are square, not circular.
        """
        bx, by = self._get_bucket(x, y)
        
        # Maximum bucket radius to search
        max_bucket_radius = (max_radius + self.bucket_size - 1) // self.bucket_size
        
        best_dist = max_radius  # Chebyshev distance
        best_food = None
        
        # Search in expanding rings of buckets for early termination
        for ring in range(max_bucket_radius + 1):
            # Early termination: if best food found and the ring's minimum 
            # possible distance is greater than our best, we're done
            if best_food is not None:
                ring_min_dist = max(0, (ring - 1) * self.bucket_size)
                if ring_min_dist >= best_dist:
                    break
            
            # Search all buckets at this ring distance
            for dbx in range(-ring, ring + 1):
                for dby in range(-ring, ring + 1):
                    # Only check perimeter of the ring (not interior, already checked)
                    if ring == 0 or abs(dbx) == ring or abs(dby) == ring:
                        check_bx = bx + dbx
                        check_by = by + dby
                        
                        # Bounds check for bucket indices
                        if 0 <= check_bx < self.buckets_x and 0 <= check_by < self.buckets_y:
                            # Check all food in this bucket
                            for fx, fy in self.buckets[check_bx][check_by]:
                                # Use Chebyshev distance to match spiral search
                                dist = max(abs(fx - x), abs(fy - y))
                                if dist < best_dist:
                                    best_dist = dist
                                    best_food = [fx, fy]
                                    # Very early exit: if food is at same position
                                    if dist == 0:
                                        return best_food
        
        return best_food if best_food else [-1, -1]
    
    def find_any_in_range(self, x, y, max_radius):
        """
        Find ANY food within max_radius (not necessarily nearest).
        Returns [fx, fy] or [-1, -1] if no food found.
        
        Uses Chebyshev distance to match original spiral search behavior.
        """
        bx, by = self._get_bucket(x, y)
        
        # Maximum bucket radius to search
        max_bucket_radius = (max_radius + self.bucket_size - 1) // self.bucket_size
        
        # Search buckets in expanding rings, return first food found in range
        for ring in range(max_bucket_radius + 1):
            for dbx in range(-ring, ring + 1):
                for dby in range(-ring, ring + 1):
                    # Only check perimeter of the ring (not interior)
                    if ring == 0 or abs(dbx) == ring or abs(dby) == ring:
                        check_bx = bx + dbx
                        check_by = by + dby
                        
                        if 0 <= check_bx < self.buckets_x and 0 <= check_by < self.buckets_y:
                            for fx, fy in self.buckets[check_bx][check_by]:
                                # Chebyshev distance matches spiral ring search
                                dist = max(abs(fx - x), abs(fy - y))
                                if dist < max_radius:
                                    return [fx, fy]
        
        return [-1, -1]
    
    def clear(self):
        """Clear all food from the spatial index"""
        for bx in range(self.buckets_x):
            for by in range(self.buckets_y):
                self.buckets[bx][by].clear()


class WorldGrid:
    def __init__(self, width, height):
        """ a grid to hold stuff like food, walls, etc """
        self.width = width
        self.height = height
        self.grid = []
        for i in range(width):
            row = []
            for j in range(height):
                row.append([])
            self.grid.append(row)
        # Track active cells for O(1) listing instead of O(width*height)
        self.active_cells = set()
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
        self.active_cells.discard((x, y))
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
            self.active_cells.discard((x, y))
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
        self.active_cells.add((x, y))
    
    def SetVal(self, x, y, val):
        if x < 0 or x >= len(self.grid):
            return
        if y < 0 or y >= len(self.grid[x]):
            return
        self.grid[x][y] = val
        # Track active cells
        if val and val != [] and val != 0:
            self.active_cells.add((x, y))
        else:
            self.active_cells.discard((x, y))
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
        self.active_cells.clear()
        return
    
    def listActive(self):
        """ Return list of active cells - O(active_count) instead of O(width*height) """
        return [[x, y, self.grid[x][y]] for x, y in self.active_cells]
    
    def sumValues(self):
        """ Return total sum of all values in active cells (for counting stacked items) """
        total = 0
        for x, y in self.active_cells:
            val = self.grid[x][y]
            if val not in ([], None, False):
                total += val
        return total

class AntColony:
    def __init__(self, _screenSize, _maxAnts, _tileSize):
        self.maxAnts = _maxAnts
        self.ants = []
        #hive pos is 80percent in the corner right
        self.screenSize = _screenSize
        self.FollowNextAnt = False
        self.brainDebugEnabled = False  # Track if brain debug mode is toggled on
        #40 pixels per tile
        self.TileSize = _tileSize
        GridSize = [int(self.screenSize[0]/self.TileSize), int(self.screenSize[1]/self.TileSize)]
        
        self.foodGrid = WorldGrid(GridSize[0], GridSize[1])
        self.terrainGrid = WorldGrid(GridSize[0], GridSize[1])
        # Dual pheromone system: separate grids for nest-seeking and food-seeking pheromones
        self.nestPheromoneGrid = WorldGrid(GridSize[0], GridSize[1])  # Dropped by ants NOT carrying food (paths to nest)
        self.foodPheromoneGrid = WorldGrid(GridSize[0], GridSize[1])  # Dropped by ants carrying food (paths to food)
        self.width = GridSize[0]
        self.height = GridSize[1]
        
        # Spatial index for O(1) food lookup instead of O(r²) spiral search
        self.foodSpatialIndex = SpatialFoodIndex(GridSize[0], GridSize[1], bucket_size=10)
        
        # Calculate field scale factor for food quantities
        # Base reference: 90x90 grid = 8100 tiles (typical for 1000x1000 screen with 11px tiles)
        self.fieldArea = self.width * self.height
        self.fieldScaleFactor = self.fieldArea / 8100.0
        
        # Scale maxFood based on field size
        self.maxFood = int(4000 * self.fieldScaleFactor)
        self.maxFood = max(100, self.maxFood)  # Minimum 100 food
        
        # Food search radius scales with grid size - smaller grids use smaller radius
        # Base: radius 10 for 90x90 grid, scale down for smaller grids
        minDimension = min(self.width, self.height)
        # self.foodSearchRadius = max(3, min(10, minDimension // 9))
        self.foodSearchRadius = 10  # Keep it constant for now
        
        # Max terrain density - higher = blocks take longer to dig through
        # At 0.5 reduction per ant, max density of 50 requires 100 ants to fully clear
        self.maxTerrainDensity = 150
        
        # Special wall value - impassable terrain that can't be dug through
        self.WALL_DENSITY = -1
        
        print(f'Field size: {self.width}x{self.height} = {self.fieldArea} tiles, scale factor: {self.fieldScaleFactor:.2f}, maxFood: {self.maxFood}, searchRadius: {self.foodSearchRadius}')

        # self.hivePos = [int(GridSize[0]*0.5), int(GridSize[1]*0.5)]
        self.hivePos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
        #the upper right corner is the tricky spot to lets put it there
        # self.hivePos = [int(GridSize[0]*0.1), int(GridSize[1]*0.1)]
        
        self.StartTime = time.time()
        
        self.BestAnts = []
        self.LastBestAnts = []
        
        # Load-mode tracking for reintroducing loaded ants
        self.loadMode = False
        self.loadedAntBrains = []
        self.loadedAntInjectChance = 0.25
        
        self.TopPheromone = 0

        self.LastSave = time.time()
        
        self.UpdateTime = 0
        
        self.totalCreatedAnts = 0
        self.totalDeadAnts = 0
        self.topFoodFound = 0
        self.totalSteps = 0
        
        # Stagnation detection variables
        self.lastLeaderboardChangeStep = 0  # Step when leaderboard last changed
        self.stagnationThreshold = 100000  # Steps without improvement before triggering evolution adjusters
        self.lastTopFitness = 0  # Track the best fitness achieved
        self.lastLowestLeaderboardFitness = 0  # Track the lowest fitness on the leaderboard
        
        # Battery level tracking for Pi mode
        self.batteryLevel = 0
        self.lastBatteryUpdate = 0
        self.batteryUpdateInterval = 10  # Update every 10 seconds
        
        # Path mode flag - only store ant history when enabled
        self.pathMode = False
        self.foodEatenPositions = []  # Track where food was picked up (for path mode visualization)
        
        # World reset settings - regenerate terrain and move hive periodically
        self.worldResetInterval = 0  # Steps between world resets (0 = disabled)
        self.lastWorldReset = 0  # Step when world was last reset
        
        # Cache fonts for performance (avoid creating font objects in draw loops)
        self._font_tiny = None   # Size 12 - for food count labels
        self._font_small = None  # Size 20 - for debug info
        self._font_medium = None # Size 26 - for FPS display
        self._font_mono = None   # Monospace size 12 - for stats
        
        # Timeline overlay system for fitness tracking
        self.fitnessHistory = []  # List of fitness snapshots over time (saved when saveData is called)
        
        # Hover info display tracking (click to show, auto-hide after timeout)
        self.hoverClickTime = 0  # Timestamp of last mouse click for hover
        self.hoverTimeout = 3.0  # Seconds to show hover info after click
        
        self.create_world()
        
                 
    def _get_font(self, size):
        """Get cached font, lazily initializing if needed"""
        if size == 12:
            if self._font_tiny is None:
                self._font_tiny = pygame.font.Font(None, 12)
            return self._font_tiny
        elif size == 20:
            if self._font_small is None:
                self._font_small = pygame.font.Font(None, 20)
            return self._font_small
        elif size == 26:
            if self._font_medium is None:
                self._font_medium = pygame.font.Font(None, 26)
            return self._font_medium
        elif size == 'mono':
            if self._font_mono is None:
                self._font_mono = pygame.font.SysFont('monospace', 12)
            return self._font_mono
        else:
            # Fallback for unknown sizes
            return pygame.font.Font(None, size)
    
    def WorldToScreen(self, apos):
        ax = apos[0]/self.width * self.screenSize[0]
        ay = apos[1]/self.height * self.screenSize[1]
        return (ax, ay)

    def ScreenToWorld(self, screen_pos):
        """Convert screen coordinates to world grid coordinates"""
        wx = int(screen_pos[0] / self.screenSize[0] * self.width)
        wy = int(screen_pos[1] / self.screenSize[1] * self.height)
        # Clamp to grid bounds
        wx = max(0, min(wx, self.width - 1))
        wy = max(0, min(wy, self.height - 1))
        return (wx, wy)

    def toggleBrainDebug(self):
        """Toggle brain debug mode. Only one ant can be in debug at a time."""
        # Toggle the enabled state
        self.brainDebugEnabled = not self.brainDebugEnabled
        
        if not self.brainDebugEnabled:
            # Turn off debug mode for any ant that has it
            for ant in self.ants:
                if ant.DebugBrain:
                    ant.DebugBrain = False
            print("Brain debug mode: OFF")
        else:
            # Turn on debug mode - pick an ant
            self.selectNewDebugAnt()
    
    def selectNewDebugAnt(self):
        """Select a new ant for brain debugging (used when debug mode is enabled)."""
        if len(self.ants) > 0:
            chosen_ant = random.choice(self.ants)
            chosen_ant.DebugBrain = True
            print(f"Brain debug: Ant ID {chosen_ant.antID[0]}")
        else:
            print("No ants available for brain debug")

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
            ant.create_brain()  # Uses global MIN_BRAIN_SIZE and MAX_BRAIN_SIZE
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

        if pos:
            foodX = pos[0]
            foodY = pos[1]
            
        # SKIP IF OUTSIDE BOUNDS
        if foodX < 0 or foodX >= self.width:
            return
        if foodY < 0 or foodY >= self.height:
            return

        # Don't place food on high-density terrain (density > half max)
        terrain_density = self.terrainGrid.GetVal(foodX, foodY)
        
        # Don't place food on rock blocks (value -1)
        if terrain_density == -1:
            return  # Rock block, no food here
        
        if terrain_density not in ([], False, None) and terrain_density > self.maxTerrainDensity / 3:
            return  # Too dense, no food here
        
        # Don't place food near the hive
        distToHive = math.sqrt((foodX - self.hivePos[0])**2 + (foodY - self.hivePos[1])**2)
        if distToHive < 15:
            return

        # Check if this is a new food location (not already in grid)
        was_empty = self.foodGrid.GetVal(foodX, foodY) in ([], 0, False)
        
        # Use IncrementVal to stack food instead of SetVal
        self.foodGrid.IncrementVal(foodX, foodY, amount)
        
        # Add to spatial index if this is a new food location
        if was_empty:
            self.foodSpatialIndex.add(foodX, foodY)
 
    def set_terrain(self, x, y, density):
        """Set terrain density at a position (0 to maxTerrainDensity scale)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Clamp density to valid range
            density = max(0, min(self.maxTerrainDensity, density))
            if density > 0:
                self.terrainGrid.SetVal(x, y, density)
            else:
                self.terrainGrid.RemoveVal(x, y)

    def _generate_terrain(self):
        """Generate terrain with patches of similar density using multi-octave noise.
        
        Creates natural-looking terrain with large coherent regions, keeping area 
        around the hive clear. Used by both create_world() and reset_world().
        """
        max_d = self.maxTerrainDensity
        
        # Terrain patch size - lower frequency = larger patches
        # 0.09 means patches roughly 11 tiles wide
        patch_frequency = 0.1
        
        # Random phase offsets for this world (makes each world unique)
        phase_x = random.random() * 1000
        phase_y = random.random() * 1000
        
        for x in range(self.width):
            for y in range(self.height):
                # Calculate distance from hive
                dist_to_hive = math.sqrt((x - self.hivePos[0])**2 + (y - self.hivePos[1])**2)
                
                # Keep area around hive clear (radius 10)
                if dist_to_hive < 5:
                    continue  # Leave as 0 (open air)
                
                # Multi-octave noise for natural terrain patches
                # Octave 1: Large patches (primary structure)
                noise1 = math.sin((x + phase_x) * patch_frequency) * math.cos((y + phase_y) * patch_frequency)
                # Bias more empty
                noise1 -= 0.3
                
                # Octave 2: Medium detail (secondary features)
                noise2 = math.sin((x + phase_x) * patch_frequency * 2.5) * math.cos((y + phase_y) * patch_frequency * 2.5) * 0.4
                
                # Octave 3: Small detail (texture)
                noise3 = math.sin((x + phase_x) * patch_frequency * 6) * math.cos((y + phase_y) * patch_frequency * 6) * 0.15
                
                # Combine octaves (-1 to 1 range)
                combined_noise = noise1 + noise2 + noise3
                
                # Map noise to density with AMPLIFIED range to create empty and max-density regions
                # Amplitude > 1 means some regions exceed bounds and get clamped to 0 or max
                # 3.8 = more truly empty and max-density patches
                amplitude = 3.8
                density = ((combined_noise + 1.55) / 3.1 * max_d - max_d/2) * amplitude + max_d/2
                
                # Add slight random variation for texture
                density += random.gauss(0, max_d * 0.05)
                
                # Clamp to valid range - creates flat empty and max-density patches
                density = max(0, min(max_d, density))
                
                # Only set non-zero densities (saves memory in sparse grid)
                if density > 0.5:
                    self.set_terrain(x, y, int(density))

    def _generate_walls(self):
        """Generate impassable walls to force pathfinding.
        
        Creates walls that ants cannot dig through, forcing them to learn 
        navigation around obstacles rather than brute-forcing straight lines.
        Includes both a center + pattern and random wall segments.
        """
        # Clear zone around hive - don't place walls within this radius
        hive_clear_radius = 20
        
        # Place the + pattern centered on the field
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Wall dimensions - scale with field size
        arm_length = min(self.width, self.height) // 4  # Each arm is 1/4 of the smaller dimension
        wall_thickness = 2  # Walls are 2 tiles thick
        
        # Horizontal arm of the +
        for x in range(center_x - arm_length, center_x + arm_length + 1):
            for t in range(wall_thickness):
                y = center_y + t - wall_thickness // 2
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Don't place walls too close to hive
                    dist_to_hive = math.sqrt((x - self.hivePos[0])**2 + (y - self.hivePos[1])**2)
                    if dist_to_hive > hive_clear_radius:
                        self.terrainGrid.SetVal(x, y, self.WALL_DENSITY)
        
        # Vertical arm of the +
        for y in range(center_y - arm_length, center_y + arm_length + 1):
            for t in range(wall_thickness):
                x = center_x + t - wall_thickness // 2
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Don't place walls too close to hive
                    dist_to_hive = math.sqrt((x - self.hivePos[0])**2 + (y - self.hivePos[1])**2)
                    if dist_to_hive > hive_clear_radius:
                        self.terrainGrid.SetVal(x, y, self.WALL_DENSITY)
        
        # Generate random wall segments to break up straight-line paths
        num_random_walls = 30
        wall_length_percent = 0.10  # Each wall is ~10% of width or height
        
        for _ in range(num_random_walls):
            # Randomly choose horizontal or vertical orientation
            is_horizontal = random.random() < 0.5
            
            if is_horizontal:
                # Horizontal wall: long in X, thin in Y
                wall_length = int(self.width * wall_length_percent)
                wall_thickness_rand = random.randint(2, 3)
                
                # Random starting position (with margin to keep wall in bounds)
                start_x = random.randint(5, self.width - wall_length - 5)
                start_y = random.randint(5, self.height - wall_thickness_rand - 5)
                
                # Check if this wall would be too close to hive at any point
                wall_center_x = start_x + wall_length // 2
                wall_center_y = start_y + wall_thickness_rand // 2
                dist_to_hive = math.sqrt((wall_center_x - self.hivePos[0])**2 + (wall_center_y - self.hivePos[1])**2)
                
                if dist_to_hive > hive_clear_radius + wall_length // 2:
                    # Place the wall
                    for x in range(start_x, start_x + wall_length):
                        for y in range(start_y, start_y + wall_thickness_rand):
                            if 0 <= x < self.width and 0 <= y < self.height:
                                # Extra check for each tile distance to hive
                                tile_dist = math.sqrt((x - self.hivePos[0])**2 + (y - self.hivePos[1])**2)
                                if tile_dist > hive_clear_radius:
                                    self.terrainGrid.SetVal(x, y, self.WALL_DENSITY)
            else:
                # Vertical wall: thin in X, long in Y
                wall_length = int(self.height * wall_length_percent)
                wall_thickness_rand = random.randint(2, 3)
                
                # Random starting position (with margin to keep wall in bounds)
                start_x = random.randint(5, self.width - wall_thickness_rand - 5)
                start_y = random.randint(5, self.height - wall_length - 5)
                
                # Check if this wall would be too close to hive at any point
                wall_center_x = start_x + wall_thickness_rand // 2
                wall_center_y = start_y + wall_length // 2
                dist_to_hive = math.sqrt((wall_center_x - self.hivePos[0])**2 + (wall_center_y - self.hivePos[1])**2)
                
                if dist_to_hive > hive_clear_radius + wall_length // 2:
                    # Place the wall
                    for x in range(start_x, start_x + wall_thickness_rand):
                        for y in range(start_y, start_y + wall_length):
                            if 0 <= x < self.width and 0 <= y < self.height:
                                # Extra check for each tile distance to hive
                                tile_dist = math.sqrt((x - self.hivePos[0])**2 + (y - self.hivePos[1])**2)
                                if tile_dist > hive_clear_radius:
                                    self.terrainGrid.SetVal(x, y, self.WALL_DENSITY)
        
        print(f"  • Walls generated: + pattern at ({center_x}, {center_y}) + {num_random_walls} random walls")

    def create_world(self):
        """Create the world with ants and varied terrain densities"""
        # Add initial ants
        for i in range(self.maxAnts):
            self.add_ant(brain=None, startP=self.hivePos)
        
        # Generate terrain
        self._generate_terrain()
        
        # Generate impassable walls in + pattern
        self._generate_walls()
    
    def reset_world(self):
        """Reset the world: regenerate terrain, clear pheromones, move hive, reset ants"""
        print("🌍 WORLD RESET - Regenerating terrain and moving hive...")
        
        # Store old hive position for reference
        old_hive = self.hivePos.copy()
        
        # Move hive to new random position
        self.hivePos = [random.randint(10, self.width-10), random.randint(10, self.height-10)]
        print(f"  • Hive moved from {old_hive} to {self.hivePos}")
        
        # Clear all grids
        self.terrainGrid.Clear()
        self.nestPheromoneGrid.Clear()
        self.foodPheromoneGrid.Clear()
        self.foodGrid.Clear()
        self.foodSpatialIndex.clear()
        
        # Regenerate terrain using shared helper
        self._generate_terrain()
        
        # Regenerate walls
        self._generate_walls()
        
        # Reset all ants to new hive position
        for ant in self.ants:
            ant.x = self.hivePos[0] + random.random() * 3
            ant.y = self.hivePos[1] + random.random() * 3
            ant.direction = random.random() * 2 * math.pi
            ant.carryingFood = False
            ant.ClossestFood = [-1, -1]
            ant.prevCellX = -1
            ant.prevCellY = -1
            ant.foodPickupPos = None
        
        # Update reset tracking
        self.lastWorldReset = self.totalSteps
        
        print(f"  • Terrain regenerated with {len(self.terrainGrid.listActive())} blocks")
        print("✅ World reset complete!")
                
           
    def Repopulate(self):
        """repopulate the ants with the best ants"""
        # print("Repopulating")
        # currentAnts = len(self.ants)
        bestAntNum = len(self.BestAnts)
        leaderboard_not_full = bestAntNum < 100
        
        if bestAntNum == 0:
            #just make new ants
            while len(self.ants) < self.maxAnts:
                if self.loadMode and leaderboard_not_full and self.loadedAntBrains:
                    if random.random() < self.loadedAntInjectChance:
                        if self._spawn_loaded_ant():
                            continue
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
            elif bestAntScore < 5:
                probBest = .5
            elif bestAntScore < 10:
                probBest = .75
            elif bestAntScore < 15:
                probBest = .85
            elif bestAntScore < 20:
                probBest = .9
            elif bestAntScore < 25:
                probBest = .95
            else:
                probBest = .99
            
            if self.loadMode and leaderboard_not_full and self.loadedAntBrains:
                if random.random() < self.loadedAntInjectChance:
                    if self._spawn_loaded_ant():
                        continue

            if random.random() >= probBest: # if less than probBest, use evolved ants; otherwise add random
                self.add_ant(brain=None, startP=self.hivePos)
                # print("Adding random ant")
            else: # add a best ant or pick two best ants as parents
                if random.random() > 0.5: #fifty fifty chance to mutate the brain or parent
                    # print("Mutating")
                    randomInt = random.randint(0, len(self.BestAnts)-1)
                    # randomInt = random.randint(0, min(5, len(self.BestAnts)-1)) #only pick the top 5 ants
                
                    # this randInt is also a fractional of the length of the best ants
                    # Better ranked ants get more clones, but with gentler scaling
                    pickedFraction = randomInt / len(self.BestAnts)
                    antBonus = 1 - pickedFraction # 1.0 for top ant, 0.0 for bottom ant
                    # Linear scaling with small boost: top ant gets ~5 clones, bottom gets 1
                    antBonus = int(1 + antBonus * 4)  # Range: 1 to 5 clones
                    antBonus = max(1, min(antBonus, 5)) # Clamp between 1-5
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
                    # need to prevent same parent from being chosen twice
                    # If only 1 best ant, can't do crossover - just clone/mutate instead
                    if len(self.BestAnts) < 2:
                        bestPick = self.BestAnts[0]
                        newBrain = self.MutateBrain(bestPick["brain"].copy())
                        newAnt = self.add_ant(brain=newBrain, startP=self.hivePos)
                        newAnt.antID[1] = "M"
                        continue
                    
                    p1Idx = random.randint(0, len(self.BestAnts)-1)
                    p2Idx = random.randint(0, len(self.BestAnts)-1)
                    while p1Idx == p2Idx:
                        p2Idx = random.randint(0, len(self.BestAnts)-1)
                    par1 = self.BestAnts[p1Idx]["brain"].copy()
                    par2 = self.BestAnts[p2Idx]["brain"].copy()
                    #select random parts of the brain from each parent
                    # Use max length to preserve brain complexity, fill missing with random parent choice
                    newBrain = []
                    maxLen = max(len(par1), len(par2))
                    for i in range(maxLen):
                        if i < len(par1) and i < len(par2):
                            # Both parents have this synapse - pick randomly
                            if random.random() > 0.5:
                                newBrain.append(par1[i])
                            else:
                                newBrain.append(par2[i])
                        elif i < len(par1):
                            # Only par1 has this synapse
                            newBrain.append(par1[i])
                        else:
                            # Only par2 has this synapse
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
        
        # Count TOTAL food quantity (including stacked food), not just cells with food
        totalFood = self.foodGrid.sumValues()

        if totalFood >= self.maxFood:
            return # already at max food
        
        whileCount = 0
        newFoodCount = 0
        while newFoodCount < ammt: # food scarcity produces more competition for food
            whileCount += 1
            if whileCount > 100:
                print("Stuck in while loop during food replenish, breaking out")
                #PICK A NEW QUADRANT
                quadrant = random.randint(0, 3)
                break
           
            #find a random spot in the quadrant
            qLeft = int(min(quads[quadrant][0][0], quads[quadrant][3][0]))
            qRight = int(max(quads[quadrant][0][0], quads[quadrant][3][0]))
            x = random.randint(qLeft, qRight)
            qBottom = int(min(quads[quadrant][1][1], quads[quadrant][2][1]))
            qTop = int(max(quads[quadrant][1][1], quads[quadrant][2][1]))
            y = random.randint(qBottom, qTop)
            foodPos = [x, y]
            
            #create small, dense clusters of food (tighter spread, more stacking)
            cluster_spread = 2  # Smaller spread = tighter clusters
            cluster_size = 30   # Fewer positions but more stacking
            for i in range(cluster_size):
                foodPosRand = [foodPos[0] + random.randint(-cluster_spread, cluster_spread), 
                               foodPos[1] + random.randint(-cluster_spread, cluster_spread)]
                #dont drop within 20 tiles of the hive
                #MUST BE WITHIN BOUNDS OF GRID
                if foodPosRand[0] < 0 or foodPosRand[0] >= self.width:
                    continue
                if foodPosRand[1] < 0 or foodPosRand[1] >= self.height:
                    continue
                distToHive = math.sqrt((foodPosRand[0] - self.hivePos[0])**2 + (foodPosRand[1] - self.hivePos[1])**2)
               
                if distToHive > 25:
                    # Only place food on low-density terrain (density <= half max)
                    terrain_density = self.terrainGrid.GetVal(foodPosRand[0], foodPosRand[1])
                    
                    # Skip rock blocks (value -1)
                    if terrain_density == -1:
                        continue
                    
                    if terrain_density in ([], False, None):
                        terrain_density = 0
                    
                    if terrain_density <= self.maxTerrainDensity / 2:
                        # Add 2-4 food per position for denser stacking
                        stack_amount = random.randint(2, 4)
                        self.add_food(foodPosRand, amount=stack_amount)
                        newFoodCount += stack_amount
                        # Check if we've hit max food (accounting for stacking)
                        if self.foodGrid.sumValues() >= self.maxFood:
                            return
        # print('food replenished')


    def MutateBrain(self, brain):
        """mutate the brain by changing values and/or structure (add/remove synapses)"""
        if len(self.ants) == 0:
            return brain
        
        # === STRUCTURAL MUTATIONS (add/remove synapses) ===
        # Uses global MIN_BRAIN_SIZE and MAX_BRAIN_SIZE
        
        # 15% chance to add a new synapse (brain growth)
        if random.random() < 0.15 and len(brain) < MAX_BRAIN_SIZE:
            # Create a new random synapse
            src = random.random() > 0.5
            dest = random.random() > 0.5
            if src:
                selectorSrc = random.randint(0, len(self.ants[0].InputSources) - 1)
            else:
                selectorSrc = random.randint(0, len(self.ants[0].neurons) - 1)
            if dest:
                selectorDst = random.randint(0, len(self.ants[0].OutputDestinations) - 1)
            else:
                selectorDst = random.randint(0, len(self.ants[0].neurons) - 1)
            frc = random.uniform(-1, 1)
            new_synapse = (src, selectorSrc, selectorDst, frc, dest)
            
            # Insert at random position to avoid always adding at end
            insert_pos = random.randint(0, len(brain))
            brain.insert(insert_pos, new_synapse)
        
        # 10% chance to remove a synapse (brain shrinkage)
        if random.random() < 0.10 and len(brain) > MIN_BRAIN_SIZE:
            remove_idx = random.randint(0, len(brain) - 1)
            brain.pop(remove_idx)
        
        # === VALUE MUTATIONS (existing behavior) ===
        
        if len(brain) == 0:
            return brain
            
        numChanges = random.randint(1, int(len(brain) * 0.8))  # change up to 80% of the brain
        for i in range(numChanges):
            idx = random.randint(0, len(brain) - 1)
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
                # frc = random.uniform(-1, 1)
                frc += random.gauss(0, 0.2)  # small random change to the force value
                frc = max(-1, min(frc, 1))  # clamp the force value between -1 and 1
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
        self.hivePos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
        # first clear all ants and set max ants to 3000
        self.ants = []
        # self.maxAnts = 3000
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
        
        # Precompute all sensor values for all ants once per frame
        # This eliminates redundant grid lookups and trig calculations during RunBrain()
        for ant in self.ants:
            ant.precompute_sensors()
        
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
            # Check bounds - world edges and walls are true blocking
            if frontPos[0] < 0 or frontPos[0] >= self.width:
                ant.blockedFront = True
            elif frontPos[1] < 0 or frontPos[1] >= self.height:
                ant.blockedFront = True
            else:
                # Check for impassable walls (WALL_DENSITY = -1)
                terrain_val = self.terrainGrid.GetVal(frontPos[0], frontPos[1])
                if terrain_val == self.WALL_DENSITY:
                    ant.blockedFront = True
                else:
                    # Terrain is traversable but has energy cost - not a true block
                    ant.blockedFront = False
            
            # Find closest food using spatial index - O(f_local) instead of O(r²)
            closestFood = [-1,-1]

            # Only search for food if ant is NOT carrying food
            if not ant.carryingFood:
                aX = int(ant.x)
                aY = int(ant.y)
                # Use spatial index for fast nearest-food lookup
                closestFood = self.foodSpatialIndex.find_nearest(aX, aY, self.foodSearchRadius)

            # Always update ClossestFood - reset to [-1,-1] if no food nearby
            ant.ClossestFood = closestFood

            if self.isAtHive(ant):
                if ant.carryingFood:
                    ant.carryingFood = False
                    ant.life += 150  # Reward: keep ant alive to forage again
                    
                    # === TRIP COMPLETION REWARD ===
                    # Base reward for completing the round trip
                    ant.fitness += 200
                    
                    # Bonus based on how far the food was from hive (harder = more reward)
                    if ant.foodPickupPos is not None:
                        pickup_distance = math.sqrt(
                            (ant.foodPickupPos[0] - self.hivePos[0])**2 + 
                            (ant.foodPickupPos[1] - self.hivePos[1])**2
                        )
                        # 3 points per tile of distance (rewards far foraging)
                        distance_bonus = int(pickup_distance * 3)
                        ant.fitness += distance_bonus
                        ant.foodPickupPos = None  # Reset for next trip
                
            ant.RunBrain()
            ant.pDirection = float(ant.direction)
           
           
           
           # history saving - only in path mode
            if self.pathMode:
                lastknownPos = ant.posHistory[-1] if len(ant.posHistory) > 0 else [0,0]
         
                moveDist = math.sqrt((ant.x - lastknownPos[0])**2 + (ant.y - lastknownPos[1])**2)
                # limited to 1 tile per move to prevent too much history
                if abs(moveDist) > 1:
                    ant.posHistory.append([ant.x, ant.y])
                    # Limit history size to prevent memory bloat
                    if len(ant.posHistory) > 200:
                        ant.posHistory = ant.posHistory[-100:]
           ## end history saving

            ant.life -= 1

            
            #check if the ant has a closestFood value, if so detect the distance and if close enough, consume the food
            
            antClosestFood = ant.ClossestFood
            if antClosestFood != [-1,-1]:
                distToFood = math.sqrt((ant.x - antClosestFood[0])**2 + (ant.y - antClosestFood[1])**2)
                if distToFood < 1.5: #increased from 1 to handle ant at edge of cell
                    if ant.carryingFood == False: #if the ant is not carrying food force ants to return home to keep eating

                        # Use DecrementVal instead of RemoveVal to support food stacking
                        if self.foodGrid.DecrementVal(antClosestFood[0], antClosestFood[1]):
                        # print(f'consuming 1 food at {antClosestFood}')
                            # Remove from spatial index if food is now empty at this location
                            remaining_food = self.foodGrid.GetVal(antClosestFood[0], antClosestFood[1])
                            if remaining_food in ([], 0, False):
                                self.foodSpatialIndex.remove(antClosestFood[0], antClosestFood[1])
                            
                            ant.ClossestFood = [-1,-1]
                            ant.FoodConsumed += 1
                            ant.life += 150  # Keep ant alive longer to return food
                            ant.fitness += 50  # Meaningful reward for finding food
                            if ant.life > 400:
                                ant.life = 400

                            ant.carryingFood = True
                            # Record pickup position for navigation fitness
                            ant.foodPickupPos = [int(ant.x), int(ant.y)]
                            
                            # Track food eaten position for path mode visualization
                            if self.pathMode:
                                self.foodEatenPositions.append([int(ant.x), int(ant.y)])

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
            # Actions when ant moves to a new cell: terrain cost, digging, pheromones
            x_pos = int(ant.x)
            y_pos = int(ant.y)
            
            # Check if ant has moved to a new cell
            if ant.prevCellX != x_pos or ant.prevCellY != y_pos:
                
                # === TERRAIN ENERGY COST AND DIGGING ===
                terrain_density = self.terrainGrid.GetVal(x_pos, y_pos)
                if terrain_density in ([], None, False):
                    terrain_density = 0
                
                if terrain_density > 0:
                    # Apply energy cost: base(1) + density/2
                    # Note: base cost of 1 is already applied via ant.life -= 1 elsewhere
                    terrain_cost = int(terrain_density / 2)
                    ant.life -= terrain_cost
                    
                    # Dig: reduce terrain density by 0.5
                    new_density = terrain_density - 0.5
                    if new_density <= 0:
                        self.terrainGrid.RemoveVal(x_pos, y_pos)
                    else:
                        self.terrainGrid.SetVal(x_pos, y_pos, new_density)
                
                # === PHEROMONE DROPPING ===
                pheromone_amount = 1.0  # Higher amount since it's dropped less frequently
                
                if ant.carryingFood:
                    # Ant carrying food drops food pheromone (path to food)
                    getCurrentPher = self.foodPheromoneGrid.GetVal(x_pos, y_pos)
                    if getCurrentPher == [] or getCurrentPher == None:
                        getCurrentPher = 0
                    newAmmt = getCurrentPher + pheromone_amount
                    if newAmmt < MAX_PHEROMONE:
                        self.foodPheromoneGrid.SetVal(x_pos, y_pos, newAmmt)
                else:
                    # Ant not carrying food drops nest pheromone (path to nest)
                    getCurrentPher = self.nestPheromoneGrid.GetVal(x_pos, y_pos)
                    if getCurrentPher == [] or getCurrentPher == None:
                        getCurrentPher = 0
                    newAmmt = getCurrentPher + pheromone_amount
                    if newAmmt < MAX_PHEROMONE:
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
        
        # Remove dead ants using list comprehension (O(n) instead of O(n²))
        # First, separate living and dead ants in a single pass
        living_ants = []
        dead_ants = []
        for ant in self.ants:
            if ant.life <= 0:
                dead_ants.append(ant)
            else:
                living_ants.append(ant)
        
        # Replace ants list with living ants only
        self.ants = living_ants
        
        # Process dead ants
        for ant in dead_ants:
            # If debugbrain ant died, auto-select new ant if debug mode is enabled
            if ant.DebugBrain:
                if self.brainDebugEnabled:
                    self.selectNewDebugAnt()
            
            # === DEATH FITNESS CALCULATION ===
            
            # If ant died while carrying food, give partial credit for progress
            if ant.carryingFood and ant.foodPickupPos is not None:
                nav_fitness = ant.calculateNavigationFitness()
                ant.fitness += nav_fitness  # Can be positive or negative (capped)
            
            foodConsumed = ant.FoodConsumed  # This is actually "completed trips"
            antFitness = ant.fitness
            
            # Exploration bonus: reward ants that ventured far (but cap it)
            # Only if they completed at least one trip
            if foodConsumed > 0:
                exploration_bonus = min(50, int(ant.FarthestTraveled * 0.5))
                antFitness += exploration_bonus
            antBrain = ant.brain
            self.totalDeadAnts += 1
            if antFitness > 2:
                # Check if this brain already exists in BestAnts
                brain_key = tuple(tuple(gene) for gene in antBrain)
                existing_idx = None
                for idx, existing_ant in enumerate(self.BestAnts):
                    existing_brain_key = tuple(tuple(gene) for gene in existing_ant["brain"])
                    if brain_key == existing_brain_key:
                        existing_idx = idx
                        break
                
                if existing_idx is not None:
                    # Brain already exists - only replace if this one scored higher
                    if antFitness > self.BestAnts[existing_idx]["fitness"]:
                        self.BestAnts[existing_idx] = {"food":foodConsumed, "brain":antBrain, "antID":ant.antID, "fitness":antFitness}
                else:
                    # New brain - add it
                    self.BestAnts.append({"food":foodConsumed, "brain":antBrain, "antID":ant.antID, "fitness":antFitness})
                
                # Update top fitness for tracking purposes (not stagnation)
                if antFitness > self.lastTopFitness:
                    self.lastTopFitness = antFitness
        
        
        #remove old pheromone cells and decay both types
        # Batch pheromone decay every 5 frames for performance
        if self.totalSteps % 5 == 0:
            decay_rate = 0.02
            
            # Handle nest pheromones
            activeNestPhers = self.nestPheromoneGrid.listActive()
            for pher in activeNestPhers:
                if pher[2] <= 0:
                    self.nestPheromoneGrid.RemoveVal(pher[0], pher[1])
                else:
                    #decay the pheromone
                    self.nestPheromoneGrid.SetVal(pher[0], pher[1], pher[2] - decay_rate)
            
            # Handle food pheromones  
            activeFoodPhers = self.foodPheromoneGrid.listActive()
            for pher in activeFoodPhers:
                if pher[2] <= 0:
                    self.foodPheromoneGrid.RemoveVal(pher[0], pher[1])
                else:
                    #decay the pheromone
                    self.foodPheromoneGrid.SetVal(pher[0], pher[1], pher[2] - decay_rate)

        # print('pheromone updated')
        
        # Only replenish food when it drops below 50% of max
        currentFood = self.foodGrid.sumValues()
        if currentFood < self.maxFood * 0.5:
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
        
        # Check for world reset interval
        if self.worldResetInterval > 0:
            steps_since_reset = self.totalSteps - self.lastWorldReset
            if steps_since_reset >= self.worldResetInterval:
                self.reset_world()
        
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
            if self.worldResetInterval > 0:
                steps_until_reset = self.worldResetInterval - (self.totalSteps - self.lastWorldReset)
                print(f'Steps Until World Reset: {steps_until_reset}/{self.worldResetInterval}')
            print(f'Update Time: {self.UpdateTime}')
            if "probbest" in repop_result:
                print(f'Repopulate Probiability of random ant: {repop_result["probbest"]}')
            print("----------------------------------------")
            #randomize the hive pos
            # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]


        # if self.totalSteps % 10000 == 0:
            # self.hivePos = [random.randint(0, self.width), random.randint(0, self.height)]

        # print('update done')


    def LoadBestAnts(self, quantity, max_files=25):
        """load the best ants from a random selection of files
        
        Args:
            quantity: Number of ants to load
            max_files: Maximum number of JSON files to randomly select (default 5)
        """
        print("Loading Best Ants")
        bestAntsFound = []
        
        searchFolder = 'dataSave'
        
        # First, collect all valid JSON file paths
        all_json_files = []
        for root, dirs, files in os.walk(searchFolder):
            for file in files:
                # Skip history files - they have a different format
                if file.endswith('_history.json'):
                    continue
                if file.endswith('.json'):
                    all_json_files.append(f'{searchFolder}/{file}')
        
        # Randomly select a handful of files
        if len(all_json_files) > max_files:
            selected_files = random.sample(all_json_files, max_files)
            print(f'Randomly selected {max_files} of {len(all_json_files)} available files')
        else:
            selected_files = all_json_files
            print(f'Loading all {len(selected_files)} available files')
        
        # Load only the selected files
        for filepath in selected_files:
            with open(filepath, 'r') as f:
                filename = os.path.basename(filepath)
                print(f'Loading: {filename}')
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

        # Cache loaded ant brains for reuse during repopulation
        valid_loaded_brains = []
        for ant in bestAntsFound:
            antBrain = ant.get("brain", [])
            if isinstance(antBrain, list) and len(antBrain) > 0:
                if isinstance(antBrain[0], list) and len(antBrain[0]) == 5:
                    valid_loaded_brains.append(antBrain)
        self.loadedAntBrains = valid_loaded_brains
        self.loadMode = True

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
                if self._spawn_loaded_ant():
                    loadedAnts += 1
                
            print(f'Loaded {loadedAnts} best ants from file')

    def _spawn_loaded_ant(self):
        if not self.loadedAntBrains:
            return False
        randomPick = random.randint(0, len(self.loadedAntBrains) - 1)
        antBrain = self.loadedAntBrains[randomPick].copy()
        if not antBrain or not isinstance(antBrain[0], list) or len(antBrain[0]) != 5:
            return False
        new_ant = self.add_ant(brain=antBrain, startP=self.hivePos)
        new_ant.antID[1] = 'L' #loaded ant
        return True


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
                if ant.life <= 1 and len(ant.posHistory) > 1:
                    # Draw path when ant is about to die (Pi mode optimization)
                    points = [self.WorldToScreen(pos) for pos in ant.posHistory]
                    pygame.draw.lines(screen, ant.Color, False, points, 1)
        else:
            for ant in self.ants:
                if len(ant.posHistory) > 1:
                    # Draw all ant paths (regardless of food consumed)
                    # Use pygame.draw.lines for better performance (single draw call per ant)
                    points = [self.WorldToScreen(pos) for pos in ant.posHistory]
                    pygame.draw.lines(screen, ant.Color, False, points, 1)
        
        # Draw white '+' markers where food was eaten (draw once, then clear so they fade)
        for pos in self.foodEatenPositions:
            screen_pos = self.WorldToScreen(pos)
            px, py = int(screen_pos[0]), int(screen_pos[1])
            # Draw small '+' shape (5 pixels each direction)
            pygame.draw.line(screen, (255, 255, 255), (px - 3, py), (px + 3, py), 1)
            pygame.draw.line(screen, (255, 255, 255), (px, py - 3), (px, py + 3), 1)
        self.foodEatenPositions.clear()  # Clear after drawing so they fade out naturally
                        
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

        # Cache active pheromone lists (avoid calling listActive() 4x per frame)
        activeNestPhers = self.nestPheromoneGrid.listActive()
        activeFoodPhers = self.foodPheromoneGrid.listActive()
        
        # Find maximum pheromone values for both types for proper scaling
        maxNestPher = 0
        maxFoodPher = 0
        
        for pher in activeNestPhers:
            pherVal = pher[2]
            if pherVal > maxNestPher:
                maxNestPher = pherVal
                
        for pher in activeFoodPhers:
            pherVal = pher[2]
            if pherVal > maxFoodPher:
                maxFoodPher = pherVal
            
        self.TopPheromone = max(maxNestPher, maxFoodPher)
        
        # Draw nest pheromones (blue - paths to nest)
        for pher in activeNestPhers:
            pherVal = pher[2]
            if pherVal <= 0.2:  # Skip barely-visible pheromones
                continue
            ppxy = self.WorldToScreen([pher[0], pher[1]])
            
            if maxNestPher > 0:
                colorVal = pherVal / maxNestPher * 255.0
                if colorVal < 10:
                    colorVal = 10
                
                ppxy = (int(ppxy[0]), int(ppxy[1]))
                pygame.draw.rect(screen, (0, 0, colorVal), ((ppxy[0]), (ppxy[1]), int(self.TileSize), int(self.TileSize)))
        
        # Draw food pheromones (red - paths to food)
        for pher in activeFoodPhers:
            pherVal = pher[2]
            if pherVal <= 0.2:  # Skip barely-visible pheromones
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

        # Draw terrain - simple cyberpunk style (cool gray + cyan accent)
        for terrain in self.terrainGrid.listActive():
            tpxy = self.WorldToScreen(terrain)
            tx, ty = int(tpxy[0]), int(tpxy[1])
            ts = int(self.TileSize)
            
            density = terrain[2]
            
            # Draw impassable walls in a cold grey color
            if density == self.WALL_DENSITY:
                # Dark cold grey base
                pygame.draw.rect(screen, (40, 80, 88), (tx, ty, ts, ts))
                # Lighter cold grey border with blue tint
                pygame.draw.rect(screen, (100, 110, 125), (tx, ty, ts, ts), 1)
                continue
            
            if density <= 0:
                continue
            
            norm_density = min(density / self.maxTerrainDensity, 1.0)
            
            # Cool gray fill - darker = denser
            gray = int(30 + (1 - norm_density) * 40)
            cyan_tint = int(norm_density * 60)
            terrain_color = (gray, gray + cyan_tint // 3, gray + cyan_tint // 2)
            pygame.draw.rect(screen, terrain_color, (tx, ty, ts, ts))
            
            # Single cyan dot for high density blocks only
            if norm_density > 0.6:
                cx, cy = tx + ts // 2, ty + ts // 2
                dot_brightness = int(80 + norm_density * 100)
                screen.set_at((cx, cy), (0, dot_brightness, dot_brightness + 20))

        # Draw food on top of terrain
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
                    font = self._get_font(12)
                    text = font.render(str(food_count), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(int(cx), int(cy)))
                    screen.blit(text, text_rect)
            

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
                
                #draw a hollow around the ant 20 pixels
                # Note: removed delay to prevent FPS drop
                
                pygame.draw.circle(screen, (RV,GV,BV), (int(pxy[0]), int(pxy[1])) , 20, width=1)
                
                #draw the direction of the ant with a line
                pygame.draw.line(screen, (0, 0, 0), (pxy[0], pxy[1]), (pxy[0] + math.cos(ant.direction) * 20, pxy[1] + math.sin(ant.direction) * 20 ))
                
                #show the ant info next to the ant
                #direction
                font = self._get_font(20)
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
                
                
                #draw brain drawbrain - bottom half overlay
            
                num_inputs = len(ant.InputSources)
                num_neurons = len(ant.neurons)
                num_outputs = len(ant.OutputDestinations)
                
                # Overlay covers bottom half of screen
                box_height = self.screenSize[1] // 2
                box_width = self.screenSize[0]
                box_top = self.screenSize[1] - box_height
                
                # Draw semi-transparent black background
                overlay = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))  # Black with 80% opacity
                screen.blit(overlay, (0, box_top))
                
                # Larger font for readability
                font = self._get_font(20)
                font_small = self._get_font(16)
                
                # Calculate spacing based on available height
                padding = 20
                usable_height = box_height - padding * 2
                input_spacing = usable_height // num_inputs
                input_spacing = min(input_spacing, 22)  # Cap spacing
                
                # Column positions (absolute x positions)
                col_inputs = 30                           # Inputs on left
                col_neurons = self.screenSize[0] // 2     # Neurons in center
                col_outputs = self.screenSize[0] - 120    # Outputs on right
                
                # Input sensor names (fuller names now that we have space)
                input_names = [
                    "direction", "prevDir", "blockedF", "foodDir", "foodFront", 
                    "oscillate", "random", "nestPherF", "foodPherF", "closeFood",
                    "nestPherL", "nestPherR", "nestPherB", "foodPherL", "foodPherR", 
                    "foodPherB", "hiveDir", "hiveDist", "carrying",
                    "terrainF", "terrainL", "terrainR", "terrainB"
                ]
                
                # Helper function: value to red-green color (val from -1 to 1)
                def value_to_color(val):
                    # Clamp value to -1 to 1 range
                    val = max(-1, min(1, val))
                    # Map -1 to 1 => 0 to 1
                    normalized = (val + 1) / 2
                    # Red (0) -> Yellow (0.5) -> Green (1)
                    if normalized < 0.5:
                        r = 255
                        g = int(255 * (normalized * 2))
                    else:
                        r = int(255 * (1 - (normalized - 0.5) * 2))
                        g = 255
                    return (r, g, 0)
                
                # Draw inputs (left column)
                inputs_start_y = box_top + padding
                for i in range(num_inputs):
                    y_pos = inputs_start_y + i * input_spacing
                    val = ant.InputSources[i]()
                    dot_color = value_to_color(val)
                    pygame.draw.circle(screen, dot_color, (col_inputs, y_pos), 5)
                    name = input_names[i] if i < len(input_names) else f"in{i}"
                    text = font_small.render(f'{name}: {val:.2f}', True, (255, 255, 255))
                    screen.blit(text, (col_inputs + 12, y_pos - 8))
                
                # Draw neurons (center column) - vertically centered
                neuron_spacing = 40
                neurons_total_height = num_neurons * neuron_spacing
                neuron_start_y = box_top + (box_height - neurons_total_height) // 2
                for i in range(num_neurons):
                    y_pos = neuron_start_y + i * neuron_spacing
                    neuron_val = ant.neurons[i]
                    dot_color = value_to_color(neuron_val)
                    pygame.draw.circle(screen, dot_color, (col_neurons, y_pos), 8)
                    text = font.render(f'N{i}: {neuron_val:.2f}', True, (255, 255, 255))
                    screen.blit(text, (col_neurons + 15, y_pos - 10))
                
                # Draw outputs (right column) - vertically centered
                output_names = ["MOVE", "TURN"]
                output_spacing = 50
                outputs_total_height = num_outputs * output_spacing
                output_start_y = box_top + (box_height - outputs_total_height) // 2
                for i in range(num_outputs):
                    y_pos = output_start_y + i * output_spacing
                    # Outputs don't have stored values, use neutral color (yellow)
                    pygame.draw.circle(screen, (255, 255, 0), (col_outputs, y_pos), 8)
                    name = output_names[i] if i < len(output_names) else f"OUT{i}"
                    text = font.render(f'{name}', True, (255, 255, 255))
                    screen.blit(text, (col_outputs + 15, y_pos - 10))
                    
                #draw the lines between the circles (synapses) using bezier curves
                for BR in ant.brain:
                    src = BR[0]
                    srcSel = BR[1]
                    dstSel = BR[2]
                    frc = BR[3]
                    dest = BR[4]
                    
                    finalForce = frc
                    
                    # Calculate start position (source)
                    if src:  # Source is an input sensor
                        xStart = col_inputs + 7
                        src_idx = srcSel % num_inputs
                        yStart = inputs_start_y + src_idx * input_spacing
                    else:  # Source is a neuron
                        xStart = col_neurons + 10
                        src_idx = srcSel % num_neurons
                        yStart = neuron_start_y + src_idx * neuron_spacing
                        finalForce = frc * ant.neurons[src_idx]
                    
                    # Calculate end position (destination)
                    if dest:  # Destination is an output
                        xEnd = col_outputs - 10
                        dst_idx = dstSel % num_outputs
                        yEnd = output_start_y + dst_idx * output_spacing
                    else:  # Destination is a neuron
                        xEnd = col_neurons - 10
                        dst_idx = dstSel % num_neurons
                        yEnd = neuron_start_y + dst_idx * neuron_spacing
                    
                    # Color based on force direction
                    if finalForce >= 0:
                        lineColor = (0, 255, 0)  # Green for positive
                    else:
                        lineColor = (255, 0, 0)  # Red for negative
                    
                    # Line thickness based on force magnitude (clamped 1-4)
                    lineThickness = max(1, min(4, int(abs(finalForce) * 4)))
                    
                    # Draw cubic bezier curve with two control points
                    # Special handling for neuron-to-neuron connections (same column)
                    is_neuron_to_neuron = (not src) and (not dest)
                    
                    if is_neuron_to_neuron:
                        # Both in neuron column - bow outward to the right
                        ctrl_offset = 60  # Fixed offset for the bow
                        ctrl1X = col_neurons + ctrl_offset  # Bow right from source
                        ctrl1Y = yStart
                        ctrl2X = col_neurons - ctrl_offset  # Bow right into dest
                        ctrl2Y = yEnd
                    else:
                        # Normal case: different columns
                        x_dist = abs(xEnd - xStart)
                        ctrl_offset = max(30, x_dist * 0.4)  # How far the control points extend
                        direction = 1 if xEnd >= xStart else -1
                        
                        ctrl1X = xStart + direction * ctrl_offset  # Away from source
                        ctrl1Y = yStart
                        ctrl2X = xEnd - direction * ctrl_offset    # Into destination
                        ctrl2Y = yEnd
                    
                    # Generate bezier curve points (cubic bezier)
                    num_segments = 16
                    bezier_points = []
                    for t_idx in range(num_segments + 1):
                        t = t_idx / num_segments
                        # Cubic bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
                        inv_t = 1 - t
                        inv_t2 = inv_t * inv_t
                        inv_t3 = inv_t2 * inv_t
                        t2 = t * t
                        t3 = t2 * t
                        
                        bx = inv_t3 * xStart + 3 * inv_t2 * t * ctrl1X + 3 * inv_t * t2 * ctrl2X + t3 * xEnd
                        by = inv_t3 * yStart + 3 * inv_t2 * t * ctrl1Y + 3 * inv_t * t2 * ctrl2Y + t3 * yEnd
                        bezier_points.append((int(bx), int(by)))
                    
                    # Draw the bezier curve
                    if len(bezier_points) > 1:
                        pygame.draw.lines(screen, lineColor, False, bezier_points, lineThickness)

        # Draw the hive position with a cool spinning shape (draw last to avoid overlap)
        self.drawHive(screen, isPi)

        # Check mouse position to determine if overlays should be shown (skip in Pi mode)
        if not isPi:
            mouse_pos = pygame.mouse.get_pos()
            screen_width = screen.get_width()
            screen_height = screen.get_height()
            show_stats = mouse_pos[1] <= 20  # Show stats if mouse is within 20px of top
            show_timeline = mouse_pos[1] >= screen_height - 20  # Show timeline if mouse is within 20px of bottom
            show_genome = mouse_pos[0] <= 20  # Show genome if mouse is within 20px of left edge
        else:
            show_stats = False
            show_timeline = False
            show_genome = False

        if show_stats:
            #show some stats on a box in the left top corner
            dataShow = self.BestAnts
            
            if len(dataShow) > 0:
                # Calculate how many items fit per column
                row_height = 12
                padding = 15
                screen_height = screen.get_height()
                max_rows_per_column = (screen_height - padding * 2) // row_height
                max_rows_per_column = max(1, max_rows_per_column)  # At least 1 row
                
                # Calculate number of columns needed
                num_columns = (len(dataShow) + max_rows_per_column - 1) // max_rows_per_column
                column_width = 380
                
                # Draw semi-transparent dark background for legibility
                overlay_width = num_columns * column_width + 40
                overlay_height = min(len(dataShow), max_rows_per_column) * row_height + padding
                overlay = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 153))  # Black with 60% opacity
                screen.blit(overlay, (5, 5))
                
                # Type to color lookup
                ColorLookup = {"CL":(200, 0, 200), "M":(255, 200, 0), "CH":(0, 200, 0), "N":(255, 0, 0), "EM":(255, 100, 255), "HY":(100, 255, 255)}
                font = self._get_font('mono')
                
                for i, ant in enumerate(dataShow):
                    # Calculate column and row position
                    column = i // max_rows_per_column
                    row = i % max_rows_per_column
                    
                    x_offset = 10 + column * column_width
                    y_pos = 10 + row * row_height
                    
                    cloneParent = ant["antID"][2] if ant["antID"][2] != -1 else ''
                    textV = f'Food:{int(ant["food"]):03} Fit:{int(ant["fitness"]):03} ID:{ant["antID"][0]:06},{ant["antID"][1]},{cloneParent}'
                    textV = textV[:55]  # Clip text to fit column
                    
                    # Get ant color from brain
                    antColor = BrainToColor(ant["brain"])
                    antColor = (antColor[0], antColor[1], antColor[2])
                    
                    # Set text color based on ant type
                    antType = ant["antID"][1]
                    color = (255, 255, 255)
                    if antType in ColorLookup:
                        color = ColorLookup[antType]
                    
                    text = font.render(textV, True, color)
                    screen.blit(text, (x_offset + 20, y_pos))
                    pygame.draw.rect(screen, antColor, (x_offset, y_pos, 10, 10))
        
        # Draw timeline overlay if mouse is at bottom edge
        if show_timeline:
            self.drawTimelineOverlay(screen, isPi)
        
        # Draw genome overlay if mouse is at left edge
        if show_genome:
            self.drawGenomeOverlay(screen)
        
        # Draw hover info for grid cell under mouse (not in Pi mode, click to show)
        if not isPi:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]  # Left mouse button
            current_time = time.time()
            
            # Update click time if mouse is pressed
            if mouse_pressed:
                self.hoverClickTime = current_time
            
            # Only show if within timeout period after click
            time_since_click = current_time - self.hoverClickTime
            if time_since_click < self.hoverTimeout:
                screen_width = screen.get_width()
                screen_height = screen.get_height()
                
                # Only show if mouse is within screen bounds
                if 0 <= mouse_pos[0] < screen_width and 0 <= mouse_pos[1] < screen_height:
                    world_pos = self.ScreenToWorld(mouse_pos)
                    wx, wy = world_pos
                    
                    # Get cell data from all grids
                    food_val = self.foodGrid.GetVal(wx, wy)
                    food = food_val if food_val not in ([], False, None) else 0
                    
                    nest_pher_val = self.nestPheromoneGrid.GetVal(wx, wy)
                    nest_pher = nest_pher_val if nest_pher_val not in ([], False, None) else 0
                    
                    food_pher_val = self.foodPheromoneGrid.GetVal(wx, wy)
                    food_pher = food_pher_val if food_pher_val not in ([], False, None) else 0
                    
                    terrain_val = self.terrainGrid.GetVal(wx, wy)
                    if terrain_val in ([], False, None):
                        terrain = 0
                        terrain_str = "0"
                    elif terrain_val == self.WALL_DENSITY:
                        terrain_str = "WALL"
                    else:
                        terrain = terrain_val
                        terrain_str = str(terrain)
                    
                    # Build info lines
                    info_lines = [
                        f"Pos: ({wx}, {wy})",
                        f"Food: {food}",
                        f"Nest Pher: {nest_pher:.2f}",
                        f"Food Pher: {food_pher:.2f}",
                        f"Terrain: {terrain_str}",
                    ]
                    
                    # Calculate box position (offset from mouse, keep on screen)
                    font = self._get_font(20)
                    box_width = 140
                    box_height = len(info_lines) * 18 + 10
                    box_x = mouse_pos[0] + 15
                    box_y = mouse_pos[1] + 15
                    
                    # Keep box on screen
                    if box_x + box_width > screen_width:
                        box_x = mouse_pos[0] - box_width - 5
                    if box_y + box_height > screen_height:
                        box_y = mouse_pos[1] - box_height - 5
                    
                    # Draw semi-transparent background
                    overlay = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 180))
                    screen.blit(overlay, (box_x, box_y))
                    
                    # Draw info text
                    for i, line in enumerate(info_lines):
                        text = font.render(line, True, (255, 255, 255))
                        screen.blit(text, (box_x + 5, box_y + 5 + i * 18))
        
    def drawHive(self, screen, isPi=False):
        """Draw a cool spinning hive visualization"""
        # Convert hive position to screen coordinates
        hive_screen_pos = self.WorldToScreen(self.hivePos)
        hive_x = int(hive_screen_pos[0])
        hive_y = int(hive_screen_pos[1])
        
        if isPi:
            # Simple thin red circle for Pi mode - minimal rendering
            pygame.draw.circle(screen, (255, 0, 0), (hive_x, hive_y), 10, 1)
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

    def updateFitnessHistory(self):
        """Update fitness history for timeline overlay"""
        if len(self.BestAnts) == 0:
            return
        
        # Sort BestAnts by fitness (highest first)
        sortedAnts = sorted(self.BestAnts, key=lambda x: x["fitness"], reverse=True)
        
        # Take every 10th ant from the best ants (up to 20 ants for performance)
        selectedAnts = []
        for i in range(0, min(len(sortedAnts), 200), 10):  # Every 10th ant, max 20 ants
            ant = sortedAnts[i]
            antColor = BrainToColor(ant["brain"])
            selectedAnts.append({
                "fitness": ant["fitness"],
                "color": (int(antColor[0]), int(antColor[1]), int(antColor[2])),
                "step": self.totalSteps
            })
        
        # Add snapshot to history (no limit - keep all history)
        self.fitnessHistory.append({
            "step": self.totalSteps,
            "ants": selectedAnts
        })

    def drawGenomeOverlay(self, screen):
        """Draw genome visualization overlay showing top 10 ant brains as pictographic DNA"""
        if len(self.BestAnts) == 0:
            return
        
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Get top 10 ants
        top_ants = self.BestAnts[:10]
        
        # Calculate dimensions
        cell_size = 14  # Size of each cell in the grid
        col_spacing = 8  # Space between ant columns
        row_spacing = 2  # Space between synapse rows
        header_height = 20  # Space for fitness label
        
        # Find max synapses across top ants for consistent height
        max_synapses = max(len(ant["brain"]) for ant in top_ants)
        
        # Calculate overlay size
        ant_column_width = 5 * cell_size + col_spacing
        overlay_width = len(top_ants) * ant_column_width + 20
        overlay_height = header_height + max_synapses * (cell_size + row_spacing) + 20
        
        # Position overlay on left side
        overlay_x = 30
        overlay_y = (screen_height - overlay_height) // 2
        
        # Draw semi-transparent dark background
        overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
        overlay_surface.fill((0, 0, 0, 200))
        screen.blit(overlay_surface, (overlay_x, overlay_y))
        
        # Get font for labels
        font = self._get_font(12)
        
        # Draw each ant's genome
        for ant_idx, ant in enumerate(top_ants):
            brain = ant["brain"]
            fitness = ant.get("fitness", 0)
            
            # Calculate column x position
            col_x = overlay_x + 10 + ant_idx * ant_column_width
            
            # Draw fitness header
            fit_text = font.render(f'{int(fitness)}', True, (255, 255, 255))
            screen.blit(fit_text, (col_x + 15, overlay_y + 4))
            
            # Draw each synapse row
            for syn_idx, synapse in enumerate(brain):
                src = synapse[0]       # bool: True=input, False=neuron
                srcSel = synapse[1]    # int: selector for source
                dstSel = synapse[2]    # int: selector for destination
                frc = synapse[3]       # float: force/weight
                dest = synapse[4]      # bool: True=output, False=neuron
                
                row_y = overlay_y + header_height + syn_idx * (cell_size + row_spacing)
                
                # Column 1: Source type (circle: filled=input, empty=neuron)
                cx = col_x + cell_size // 2
                cy = row_y + cell_size // 2
                if src:  # Input source
                    pygame.draw.circle(screen, (100, 200, 255), (cx, cy), cell_size // 2 - 2)
                else:  # Neuron source
                    pygame.draw.circle(screen, (100, 200, 255), (cx, cy), cell_size // 2 - 2, 2)
                
                # Column 2: Source selector (colored by value)
                cx = col_x + cell_size + cell_size // 2
                # Color based on selector value (cycle through colors)
                sel_colors = [(255, 100, 100), (255, 200, 100), (255, 255, 100), 
                              (100, 255, 100), (100, 255, 200), (100, 255, 255),
                              (100, 200, 255), (100, 100, 255), (200, 100, 255), (255, 100, 200)]
                src_color = sel_colors[srcSel % len(sel_colors)]
                # Draw triangle pointing right for source
                points = [(cx - 4, cy - 5), (cx - 4, cy + 5), (cx + 5, cy)]
                pygame.draw.polygon(screen, src_color, points)
                
                # Column 3: Force/weight (bar showing magnitude and direction)
                bar_x = col_x + 2 * cell_size + 2
                bar_width = cell_size - 4
                bar_height = cell_size - 4
                bar_y = row_y + 2
                
                # Background
                pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height))
                
                # Force bar - green for positive, red for negative
                normalized_frc = max(-1, min(1, frc))  # Clamp to -1 to 1
                if normalized_frc >= 0:
                    fill_width = int(normalized_frc * bar_width)
                    pygame.draw.rect(screen, (50, 200, 50), (bar_x, bar_y, fill_width, bar_height))
                else:
                    fill_width = int(abs(normalized_frc) * bar_width)
                    pygame.draw.rect(screen, (200, 50, 50), (bar_x + bar_width - fill_width, bar_y, fill_width, bar_height))
                
                # Border
                pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 1)
                
                # Column 4: Destination selector (colored by value)
                cx = col_x + 3 * cell_size + cell_size // 2
                dst_color = sel_colors[dstSel % len(sel_colors)]
                # Draw triangle pointing left for destination
                points = [(cx + 4, cy - 5), (cx + 4, cy + 5), (cx - 5, cy)]
                pygame.draw.polygon(screen, dst_color, points)
                
                # Column 5: Destination type (square: filled=output, empty=neuron)
                sq_x = col_x + 4 * cell_size + 2
                sq_y = row_y + 2
                sq_size = cell_size - 4
                if dest:  # Output destination
                    pygame.draw.rect(screen, (255, 200, 100), (sq_x, sq_y, sq_size, sq_size))
                else:  # Neuron destination
                    pygame.draw.rect(screen, (255, 200, 100), (sq_x, sq_y, sq_size, sq_size), 2)
            
            # Draw separator line between ants (except after last)
            if ant_idx < len(top_ants) - 1:
                sep_x = col_x + 5 * cell_size + col_spacing // 2
                pygame.draw.line(screen, (60, 60, 60), 
                               (sep_x, overlay_y + header_height - 5),
                               (sep_x, overlay_y + overlay_height - 10), 1)
        
        # Draw column header labels
        label_y = overlay_y + overlay_height - 15
        labels = ["SRC", "→", "FRC", "←", "DST"]
        for i, label in enumerate(labels):
            lx = overlay_x + 10 + i * cell_size + 2
            label_surface = font.render(label, True, (150, 150, 150))
            screen.blit(label_surface, (lx, label_y))

    def drawTimelineOverlay(self, screen, isPi=False):
        """Draw fitness timeline overlay at bottom of screen"""
        if len(self.fitnessHistory) < 2:
            return
        
        screen_height = screen.get_height()
        screen_width = screen.get_width()
        
        # Timeline dimensions
        timeline_height = 120
        timeline_width = screen_width - 20
        timeline_x = 10
        timeline_y = screen_height - timeline_height - 10
        
        # Draw semi-transparent dark background
        overlay = pygame.Surface((timeline_width, timeline_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 153))  # Black with 60% opacity
        screen.blit(overlay, (timeline_x, timeline_y))
        
        # Find min/max fitness for normalization
        all_fitness_values = []
        for snapshot in self.fitnessHistory:
            for ant in snapshot["ants"]:
                all_fitness_values.append(ant["fitness"])
        
        if len(all_fitness_values) == 0:
            return
        
        min_fitness = min(all_fitness_values)
        max_fitness = max(all_fitness_values)
        fitness_range = max_fitness - min_fitness
        
        if fitness_range == 0:
            fitness_range = 1  # Prevent division by zero
        
        # Draw timeline graph
        graph_height = timeline_height - 20
        graph_width = timeline_width - 20
        graph_x = timeline_x + 10
        graph_y = timeline_y + 10
        
        # Calculate time range
        if len(self.fitnessHistory) > 1:
            time_range = self.fitnessHistory[-1]["step"] - self.fitnessHistory[0]["step"]
            if time_range == 0:
                time_range = 1
        else:
            time_range = 1
        
        # Draw fitness lines for each ant position
        for snapshot_idx, snapshot in enumerate(self.fitnessHistory):
            if snapshot_idx == 0:
                continue  # Skip first snapshot (need previous point for line)
            
            prev_snapshot = self.fitnessHistory[snapshot_idx - 1]
            
            # Calculate x positions for current and previous snapshots
            curr_time_progress = (snapshot["step"] - self.fitnessHistory[0]["step"]) / time_range
            prev_time_progress = (prev_snapshot["step"] - self.fitnessHistory[0]["step"]) / time_range
            
            curr_x = graph_x + int(curr_time_progress * graph_width)
            prev_x = graph_x + int(prev_time_progress * graph_width)
            
            # Draw lines for each ant (match by position in sorted list)
            max_ants = min(len(snapshot["ants"]), len(prev_snapshot["ants"]))
            for ant_idx in range(max_ants):
                curr_ant = snapshot["ants"][ant_idx]
                prev_ant = prev_snapshot["ants"][ant_idx]
                
                # Normalize fitness to graph coordinates
                curr_fitness_norm = (curr_ant["fitness"] - min_fitness) / fitness_range
                prev_fitness_norm = (prev_ant["fitness"] - min_fitness) / fitness_range
                
                curr_y = graph_y + graph_height - int(curr_fitness_norm * graph_height)
                prev_y = graph_y + graph_height - int(prev_fitness_norm * graph_height)
                
                # Use current ant's brain color
                color = curr_ant["color"]
                
                # Draw line segment
                if abs(curr_x - prev_x) > 0 or abs(curr_y - prev_y) > 0:
                    pygame.draw.line(screen, color, (prev_x, prev_y), (curr_x, curr_y), 2)

    def saveData(self):
        
        # Update fitness history snapshot when saving
        self.updateFitnessHistory()
        
        # Note: Duplicates are prevented at insertion time in update()
        # so no cleanup needed here
        
        #check to see if the data chenged from last time
        if self.BestAnts == self.LastBestAnts:
            # print('No changes to save')
            return
        
        self.LastBestAnts = self.BestAnts.copy()
        
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
        self.testMode = False
        self.headlessMode = False
        
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
        parser.add_argument('--load', action='store_true', help='Load best ants from saved data files')
        parser.add_argument('--test', action='store_true', help='Test mode: 1 ant, 1 second delay, brain debug enabled')
        parser.add_argument('--headless', action='store_true', help='Headless mode: no rendering, runs as fast as possible')
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

        if args.test:
            print('Test Mode: 1 ant, 1 second delay, brain debug enabled')
            self.testMode = True
            self.maxAnts = 1

        if args.headless:
            print('Headless Mode: no rendering, running as fast as possible')
            self.headlessMode = True
        
        # if len(sys.argv) > 1:
        #     print('Arguments: ', sys.argv[1])
        #     if sys.argv[1] == "pi":
        #         isPi = True
        #         print('Running on Raspberry Pi')

        tileSize = 11  #smaller is slower

        # Skip pygame initialization in headless mode
        if not self.headlessMode:
            pygame.init()
                    
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
        
        # Set path mode on colony if drawPaths is enabled
        self.antColony.pathMode = self.drawPaths
        
        # Only load best ants if --load flag is set
        if args.load:
            self.antColony.LoadBestAnts( self.maxAnts )
        else:
            print('Starting fresh (use --load to load saved ants)')

        # Enable brain debug mode in test mode
        if self.testMode:
            self.antColony.toggleBrainDebug()

        #first run update 20000 times
        lastPercent = 0
        if self.isPi == False and not self.testMode:
            newAnts = 500
            #add new ants before training
            print(f'Adding {newAnts} new random ants')
            for i in range(newAnts):
                self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            # Skip pre-training in headless mode - it will train during run
            if not self.headlessMode:
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
                if event.type == pygame.KEYDOWN:
                    # H key - reduce max ants by 50%
                    if event.key == pygame.K_h:
                        self.maxAnts = max(10, self.maxAnts // 2)
                        self.antColony.maxAnts = self.maxAnts
                        print(f'Reduced max ants to {self.maxAnts}')
                    # T key - toggle trail view mode
                    if event.key == pygame.K_t:
                        self.drawPaths = not self.drawPaths
                        self.antColony.pathMode = self.drawPaths
                        print(f'Trail view mode: {"ON" if self.drawPaths else "OFF"}', flush=True)
                    # P key - toggle brain debug mode
                    if event.key == pygame.K_p:
                        self.antColony.toggleBrainDebug()

            #run 5 times between each draw
            #not in pi mode
            if self.testMode:
                # Test mode: single update per frame
                self.antColony.update()
            elif not self.isPi:
                for i in range(5):
                    self.antColony.update()
            else:
                self.antColony.update()

            fps = self.clock.get_fps()
           

            if self.drawPaths:
            # self.antColony.drawAnts(self.screen, isPi=self.isPi)
                self.antColony.drawPaths(self.renderSurface, isPi= self.isPi)
            else:
                self.antColony.drawAnts(self.renderSurface, isPi=self.isPi)
            
            # Scale up the render surface to the screen if using pixel scaling
            if self.isPi and self.pixelScale > 1:
                # Use NEAREST neighbor scaling to maintain sharp pixels (no blurring)
                scaled_surface = pygame.transform.scale(self.renderSurface, self.screenSize)
                self.screen.blit(scaled_surface, (0, 0))
            elif self.renderSurface is not self.screen:
                # Non-Pi mode with separate render surface (shouldn't happen normally)
                self.screen.blit(self.renderSurface, (0, 0))
            
            # Draw FPS and Ants count directly on screen (not affected by scaling)
            if not self.drawPaths:
                font = self.antColony._get_font(26)
                fps_text = font.render(f'FPS: {fps:.1f}', True, (255, 255, 255))
                self.screen.blit(fps_text, (self.screenSize[0]-100, 10))
                ants_text = font.render(f'Ants: {len(self.antColony.ants)}', True, (255, 255, 255))
                self.screen.blit(ants_text, (self.screenSize[0]-100, 30))
                # Display total food on field
                total_food = self.antColony.foodGrid.sumValues()
                food_text = font.render(f'Food: {total_food}', True, (255, 255, 255))
                self.screen.blit(food_text, (self.screenSize[0]-100, 50))
            
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

            if keys[pygame.K_r]:
                self.antColony.reset_world()
                
            if keys[pygame.K_u]:
                percentDone = 0
                for i in range(1000):
                    nowPercent = int(i/1000 * 100)
                    if nowPercent != percentDone:
                        print(f'FAST TRAINING: {nowPercent}%')
                        percentDone = nowPercent
                    self.antColony.update()

            # # if you hit key R, add 1000 random ants
            # if keys[pygame.K_r]:
            #     for i in range(10000):
            #         self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)
            
            #add key will add 1000 ants
            if keys[pygame.K_a]:
                #make max ants 1000
                self.maxAnts += 1000
                self.antColony.maxAnts = self.maxAnts
                # for i in range(1000):
                #     self.antColony.add_ant(brain=None, startP=self.antColony.hivePos)

            ###
            
            # Adaptive ant count based on FPS (skip in test mode)
            if self.testMode:
                pass  # Keep maxAnts at 1 in test mode
            elif self.isPi:
                # Pi mode: target lower FPS, be more aggressive with scaling
                if fps < 3:
                    self.maxAnts -= 1
                    if self.maxAnts < 10:
                        self.maxAnts = 10
                elif fps > 8:
                    self.maxAnts += 1
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
            if self.testMode:
                self.clock.tick(1)  # 1 FPS = 1 second delay per frame
            else:
                self.clock.tick()

        pygame.quit()

    def run_headless(self):
        """Run simulation without rendering - as fast as possible"""
        print('Running in HEADLESS mode (no rendering)')
        print('Press Ctrl+C to stop and save')
        
        running = True
        last_report_time = time.time()
        updates_since_report = 0
        
        try:
            while running:
                self.antColony.update()
                updates_since_report += 1
                
                # Report progress every 5 seconds
                current_time = time.time()
                if current_time - last_report_time >= 5.0:
                    elapsed = current_time - last_report_time
                    updates_per_sec = updates_since_report / elapsed
                    
                    best_fitness = 0
                    if len(self.antColony.BestAnts) > 0:
                        best_fitness = self.antColony.BestAnts[0].get("fitness", 0)
                    
                    print(f'[Headless] Steps: {self.antColony.totalSteps} | '
                          f'Updates/sec: {updates_per_sec:.1f} | '
                          f'Ants: {len(self.antColony.ants)} | '
                          f'Leaderboard: {len(self.antColony.BestAnts)} | '
                          f'Best Fitness: {best_fitness}')
                    
                    last_report_time = current_time
                    updates_since_report = 0
                    
        except KeyboardInterrupt:
            print('\nStopping headless mode...')
        
        # Save before exit
        print('Saving data...')
        self.antColony.saveData()
        print('Done.')

if __name__ == "__main__":
    print('Starting Simulation')
    game = Game()
    if game.headlessMode:
        game.run_headless()
    else:
        game.run()
