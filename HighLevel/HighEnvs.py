import sys
import gym
from gym import spaces
import numpy as np
import pickle
import random
import CoppeliaTower
from HighTowerSim import Tile, getClusters, getTile, getRotation, tupleTransform, getHeight, getRotateMat
from stable_baselines3 import SAC
# from DQNHigh1 import Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import tensorflow as tf
from stable_baselines3.common.monitor import Monitor
from placeTileEnv import TileEnv
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


obsLen = 38


# The autoencoder. It does not work, but was left here for reference
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])

    # re-expand to 38 values
    self.decoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(obsLen, activation='linear')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class HighEnv1(gym.Env):
    def __init__(self, fileName="MinTowerSim_1000.txt"):
        self.latent_dim = 38  # 15  # for autoencoder
        self.model = SAC.load("SAC1000_TilePlace_best/best_model")  # tile placement AI
        self.fileName = fileName
        self.file = open(self.fileName, 'rb')
        # used to build tower
        self.maxTowers = 300  # how many towers to train on (too many in file, so limit)
        self.towerCount = 0
        self.oldtower = None  # placeholder, immediately changed
        self.illegalMoves = 0
        self.recordIllegal = []
        self.getNext()  # assigns tower
        # used as AI input - need to keep a copy of the current (for state) and the initial tower (for ordered simulation)
        self.modifiedtower = self.copy(self.oldtower)
        self.myPillars = self.getPillars(5)
        # check for at least 1 valid move
        while self.checkEnd():
            self.myPillars = self.getPillars(5)
            print("Re-drawing Pillars")
        self.campPillars = self.getPillars(6)
        self.partnerPillars = self.getPillars(5)
        self.card = 6
        self.count = 0

        self.obsLen = self.latent_dim  # remnant of the autoencoder
        self.state = self.getState()
        self.history = []  # keep track of pillars and tiles placed this turn and their order - for simulation

        self.observation_space = spaces.Box(-1, 6, shape=(self.obsLen,))
        self.action_space = spaces.Discrete(5)  # numbers 0-4
        self.reward_range = (-26, 26)  # approximately

    # make a completely new copy of the given tower
    def copy(self, tower):
        newTower = []
        try:
            for level in tower:
                floor = []
                for tile in level:
                    newTile = Tile(tile.id, tile.spots, rotation=tile.rotation,
                                   origin=tile.origin, outline=tile.outline, colours=tile.colours)
                    newTile.worldSpots = np.copy(tile.worldSpots)
                    newTile.filled = np.copy(tile.filled)
                    floor.append(newTile)
                newTower.append(floor)
            return newTower
        except Exception as e:
            print(self.towerCount)

    def getNext(self):
        try:
            if self.towerCount >= self.maxTowers:
                self.file.close()
                self.file = open(self.fileName, 'rb')
                self.towerCount = 0
            self.towerCount += 1
            self.oldtower = pickle.load(self.file)[0]

        except EOFError:
            self.file.close()  # close file, reopen, try again
            self.file = open(self.fileName, 'rb')
            self.oldtower = pickle.load(self.file)[0]

    # generate random array of num length, with numbers 1-5 according to probability dist
    # 0 = yellow (freq = 39%)
    # 1 = red (freq = 26%)
    # 2 = black (freq = 16%)
    # 3 = white (freq = 11%)
    # 4 = blue (freq = 8%)
    def getPillars(self, num):
        choices = [1, 2, 3, 4, 5]
        freq = [0.39, 0.26, 0.16, 0.11, 0.08]
        pillars = random.choices(choices, freq, k=num)
        pillars.sort()
        return pillars


    def getState(self):
        index = 0
        ids = ["Tile01", "Tile02", "Tile03", "Tile04", "Tile05", "Tile06", "Tile07", "Tile08", "Tile09", "Tile10",
               "Tile11", "Tile12", "Tile13", "Tile14", "Tile15", "Tile16", "Tile17", "Tile18"]
        state = np.zeros(obsLen)
        floorCount = 0
        prevRot = 0
        prevOri = [0, 0, 0] # origin of 1st tile is [0,0,0]
        try:
            for floor in self.modifiedtower:  # should only have 1 item in each
                for tile in floor:
                    id = ids.index(tile.id)
                    state[index] = id
                    index += 1
                    # tile.origin is with respect to the origin of the tile underneath
                    tempOri = [x-y for x, y in zip(tile.origin, prevOri)]
                    state[index:index+3] = [round(i/500, 2) for i in tempOri]
                    index += 3
                    # same for rotation
                    temp = np.deg2rad(tile.rotation - prevRot)
                    state[index] = np.sin(temp)
                    index += 1
                    state[index] = np.cos(temp)
                    index += 1

                    prevRot = tile.rotation
                    prevOri = tile.origin
                    count = 0  # make sure there are always 5 spots
                    for spot in tile.spots:  # use local spots
                        if tile.filled[count]:
                            state[index] = -1  # filled, otherwise provide colour
                        else:
                            state[index] = tile.colours[count]
                        index += 1
                        count += 1

                    for i in range(count, 5):
                        state[index] = 6  # ghost spots are 6
                        index += 1
                floorCount += 1

            for _ in range(floorCount, 3):
                index += 11

            # now add pillars
            state[index:index+5] = self.myPillars
            index += 5

        except Exception as e:
            print(e)

        # this is where the state would be transformed by the encoder to provide a compressed form
        return state


    def getCoord(self, action):
        # action is 0-4 representing location of pillars on top floor (order is set)
        tile = self.modifiedtower[-1][0]
        # check that this tile has that many actions. if not, return -1
        if len(tile.spots) <= action:
            return None

        coord = np.copy(tile.spots[action])  # avoid alliasing
        filled = tile.filled[action]
        colour = tile.colours[action]
        if filled:
            return None  # spot is filled, illegal action, state won't change, try again

        if colour not in self.myPillars:
            return None  # don't have that colour pillar
        else:
            # get index of colour
            for i in range(5):
                if self.myPillars[i] == colour:
                    self.myPillars[i] = self.getPillars(1)[0]  # replace w new colour (or false colour)
                    break  # exit loop
        return coord

    def getReward(self, collapse):
        if collapse:
            print("Failed!")
            return -10
        else:
            return 15

    # just gets x and  y coords of pillars and adds id of new tile
    # needed when adding a tile using the AI
    def getTileState(self, id):
        tile = self.modifiedtower[-1][0]
        state = np.zeros(11)
        index = 0
        for spot in tile.spots:
            state[index:index + 2] = spot
            index += 2
        state[10] = id
        state = [x / 50 for x in state]
        return state


    # first checks if all spots on the top layer are filled
    # if so, check if numLevels < 3
    # if numLevels = 3, just return endflag = true
    # otherwise add tile to sim. Try 5 times with noise
    def addFloor(self):
        tile = self.modifiedtower[-1][0]
        filled = tile.filled
        if 0 in filled:
            return False, 0  # still spots left, no additional floor

        numLevels = len(self.modifiedtower)
        if numLevels == 3:
            return True, 10  # finished all 3 tiles, end episode without adding next tile

        # select new tile, try 3 times w diff tiles
        for a in range(3):
            tileIndex = random.randint(0, 17)
            newTile = getTile(tileIndex)

            obs2 = self.getTileState(tileIndex)
            action2, _states2 = self.model.predict(obs2, deterministic=True)
            newTile.worldSpots = []
            # add to tower
            newTile.origin = [action2[0], action2[1], getHeight(numLevels)]
            newTile.rotation = action2[2]
            rotate = getRotateMat(newTile.rotation)
            for spot in newTile.spots:
                tempSpot = tupleTransform(spot, newTile.origin, rotate)
                newTile.worldSpots.append(np.append(tempSpot, getHeight(numLevels)))
                tempHistory = self.history + [[1, newTile]]
                # simulate old tower with addons

            collapse = CoppeliaTower.simulate(self.oldtower, tempHistory)
            # if not simulating, uncomment next line and comment out prev
            # collapse = 0  # didn't collapse
            if not collapse:
                self.history = tempHistory
                self.modifiedtower.append([newTile])
                return False, 15  # added, keep going (additional reward for risky move), exits loop

        # otherwise give up
        return True, 5  # end episode, give minimal reward, it's not the fault of the pillar placement agent

    # based on state, check if any legal moves left. if so, return 0, else return 1
    def checkEnd(self):
        tile = self.modifiedtower[-1][0]
        count = 0
        for spot in tile.spots:
            # check if spot is empty and colour is available
            if not tile.filled[count] and tile.colours[count] in self.myPillars:
                # at least 1 available spot
                return False
            count += 1
        return True  # no more spots


    def step(self, action):
        location = self.getCoord(action)  # get coordinate of chosen spot - must be local
        self.count += 1
        endflag = 0
        tempReward = 0

        if location is None:  # not valid move
            print("Illegal!")
            self.illegalMoves += 1
            reward = -15  # negative reward, no change in state, return
        else:  # only modify tower if mover is valid/successful
            tile = self.modifiedtower[-1][0]
            self.history.append([0, location, tile.id])  # 0 means it's a pillar object

            # build tower so that it stands, then add pillar. Observe collapse
            collapse = CoppeliaTower.simulate(self.oldtower, self.history)

            # if not simulating: uncomment next line
            # collapse = 0  # didn't collapse, comment out previous

            if collapse == -1:
                print("Error in sim!")
                quit()

            self.modifiedtower[-1][0].filled[action] = 1
            self.card -= 1
            if collapse or self.card == 0:
                endflag = 1
            else:
                endflag, tempReward = self.addFloor()  # if finished 3rd level, success (endFlag = 1) else add next tile
                # Note: if you are not simulating, modify self.addFloor to always be successful

            # temp reward gives reward for cases where ending without card == 0
            reward = self.getReward(collapse)
            reward += tempReward  # from adding floor
             # finished episode

            # if not simulating, uncomment next line
            # reward = 15  # overwrite to simplify

        info = dict()
        self.state = self.getState()
        if self.checkEnd():
            endflag = 1

        if endflag:
            self.count = 0
        if self.count > 20 and endflag == 0:
            endflag = 1  # give up
            print("Give up!!")


        print("Reward:", reward)
        print()
        return self.state, reward, endflag, info

    def getIllegal(self):
        return self.recordIllegal

    def reset(self):
        self.getNext()  # assigns tower

        self.modifiedtower = self.copy(self.oldtower)
        # new set of pillars
        self.myPillars = self.getPillars(5)

        while self.checkEnd():
            self.myPillars = self.getPillars(5)
            print("Re-drawing Pillars")

        self.card = 6
        self.count = 0
        self.history = []
        self.recordIllegal.append(self.illegalMoves)
        self.illegalMoves = 0
        self.state = self.getState()
        return np.float32(self.state)

