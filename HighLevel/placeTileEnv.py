# environment for placing tile AI
import sys

import gym
from gym import spaces
import numpy as np
import pickle
import random
import PlaceTileSim
from HighTowerSim import Tile, getClusters, getTile, getRotation, tupleTransform, getHeight, getRotateMat

class TileEnv(gym.Env):
    def __init__(self, fileName, writeFile):
        self.writeFile = writeFile
        if self.writeFile:
            self.file = open(fileName, 'wb')
        # used to build tower
        self.towerCount = 0
        self.tower = None
        # tile ids of 1st and 2nd floors
        self.firstTile = random.randint(0, 17)
        self.secondTile = random.randint(0, 17)
        # pillars in local coords = 2 * 5 = 10
        # ID of 2nd floor = 1. total = 11
        self.obsLen = 11
        self.state = self.getState()

        self.observation_space = spaces.Box(-2, 2, shape=(self.obsLen,))
        self.action_space = spaces.Box(-180, 180, shape=(3,))
        self.reward_range = (-60, 10)

    def getState(self):
        index = 0
        state = np.zeros(self.obsLen)
        tile = getTile(self.firstTile)
        length = len(tile.spots)
        for i in range(length):  # all spots filled
            tile.filled[i] = 1
        self.tower = [[tile], [getTile(self.secondTile)]]

        for spot in tile.spots:
            state[index:index+2] = spot
            index += 2
        # leave ghost spots as 0s
        index = 10
        state[index] = self.secondTile
        state = [x/50 for x in state]
        return state


    def getDist(self, action):
        dist = np.linalg.norm(action)
        return dist

    def step(self, action):
        # action should be x, y, theta of new tile
        firstOrigin = self.tower[0][0].origin
        self.tower[1][0].origin = [action[0]+firstOrigin[0], action[1]+firstOrigin[1], 52] # height can be hardcoded for this
        self.tower[1][0].rotation = action[2]  # rotation of 1st tile is 0
        # build tower based on state and action
        collapsed = PlaceTileSim.simulate(self.tower)

        # if collapsed, reward is distance from local origin (encourages towards right area)
        if collapsed:
            dist = -self.getDist(action[0:2])
            reward = dist/10

        # otherwise reward is flat positive value
        else:
            reward = 10
            # save model
            if self.writeFile:
                try:
                    pickle.dump([self.tower], self.file) # assume open
                    if self.towerCount > 25000:
                        self.file.close()
                except Exception as e:
                    print(self.towerCount)

        endflag = 1   # no matter what happens endflag is always 1
        info = dict()
        return np.array(self.state), reward, endflag, info


    def reset(self):

        self.towerCount += 1
        self.tower = None
        # tile ids of 1st and 2nd floors
        self.firstTile = random.randint(0, 17)
        self.secondTile = random.randint(0, 17)
        self.state = self.getState()

        return self.state


