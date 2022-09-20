# code for algorithmic agents: level difficulty agent and colour selector agent
import pickle
import numpy as np
from HighTowerSim import Tile
import random

def getPillars(num):
    choices = [1, 2, 3, 4, 5]
    freq = [0.39, 0.26, 0.16, 0.11, 0.08]
    pillars = random.choices(choices, freq, k=num)
    pillars.sort()
    return pillars

# based on the open spot values (extracted from tower) and the pillars, figure out how many legal moves
def levelDiff(tower, pillars):
    tile = tower[-1][0]
    filled = tile.filled
    num = len(filled)
    colours = [tile.colours[j] for j in range(num) if filled[j]==0]
    count = 0  # keeps track of number of legal moves
    for i in range(len(colours)):
        if colours[i] in pillars:
            count += 1
            pillars.remove(colours[i])  # remove to avoid double counting

    if count >= 3:
        return 3
    elif count == 2:
        return 2
    else:
        return 1

def colourSelector(tower, pillars, camp):
    tile = tower[-1][0]
    filled = tile.filled
    num = len(filled)
    colours = [tile.colours[j] for j in range(num) if filled[j] == 0]
    count = 0
    pool = pillars + camp
    newPillars = []
    for i in range(len(colours)):
        if colours[i] in pool:
            count += 1
            newPillars.append(colours[i])
            pool.remove(colours[i])

    # select pillars from pool in order of frequency and variation
    randCamp = random.sample(pool, k=5-count)
    newPillars += randCamp
    try:
        for i in randCamp:
            pool.remove(i)
    except Exception as e:
        print(e)
    return newPillars, pool



fileName = "MinTowerSim_1000.txt"
file = open(fileName, 'rb')
for i in range(10):
    tower = pickle.load(file)[0]
    pillars = getPillars(5)
    camp = getPillars(6)

    newPillars, newCamp = colourSelector(tower, pillars, camp)
    level = levelDiff(tower, newPillars)
