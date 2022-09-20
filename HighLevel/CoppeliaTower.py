
import sim
from HighTowerSim import Tile
import SawyerSim
import sys
import numpy as np
import quaternion
import time
import pickle
import random


# pt1 has [x,y], pt2 has [x,y,z]. Ignore z
def getDist(pt1, pt2):
    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return dist
# tower is the only one we really care about
# others are just to check that tower is viable
# numTowers = 1
# fileName = "MinTowerSim_" + str(numTowers) + ".txt"
# f = open(fileName, 'rb')
# temp = pickle.load(f)
# tower = temp[0]
# f.close()

# tower has up to 3 elements, each with a tile
# tiles have information on origin, rotation, spots (in world coords), and filled
# tiles can be placed at exact location, pillars must be placed at location + 4.5e-2
# addons is a list of things to be built after tower is done, in a given order. this includes pillars and tiles
def simulate(tower, addons=None):
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to CoppeliaSim
    pillarHandles = []  # save all generated pillar handles
    pillarLocations = []  # save all locations the pillars should be at to double check (just x and y)
    parentTiles = []  # need to save the parent tile for pillars
    if clientID!=-1:
        # Now try to retrieve data in a blocking fashion (i.e. a service call):

        # sim.simxSetBooleanParameter(clientID, sim.sim_boolparam_display_enabled, False, sim.simx_opmode_blocking)  # disable gpu
        res, objs=sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print ('Remote API function call returned with error code: ', res)
            sys.quit()

        pillarHandle = sim.simxGetObjectHandle(clientID, "Pillar", sim.simx_opmode_blocking)[1]
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        for level in tower:
            for tile in level:
                newestTile = tile
                # for each tile, place at spot
                # get handle for tile
                handle = sim.simxGetObjectHandle(clientID, tile.id, sim.simx_opmode_blocking)[1]
                newTile = sim.simxCopyPasteObjects(clientID, [handle], sim.simx_opmode_blocking)[1][0]
                # position
                # print(tile.id)
                origin = [x/1000 for x in tile.origin] # convert from mm to m
                # print(origin)
                origin = [i*5 for i in origin] # multiply by 5 because of coppelia scaling
                origin[2] *= 0.7 # lower it a bit
                # set rotation first
                ori = sim.simxGetObjectQuaternion(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
                oriQ = np.quaternion(ori[3], ori[0], ori[1], ori[2])
                rot = tile.rotation
                newQ = np.quaternion(np.cos(rot / 2), 0, 0, np.sin(rot / 2))
                quat = newQ * oriQ
                quat2 = [quat.x, quat.y, quat.z, quat.w]
                sim.simxSetObjectQuaternion(clientID, newTile, -1, quat2, sim.simx_opmode_blocking)
                sim.simxSetObjectPosition(clientID, newTile, -1, origin, sim.simx_opmode_blocking)
                # rotation
                # for each tile, fill pillars
                # time.sleep(1)
                sim_handle_parent = newTile
                # print("Pillars")
                # order pillars by distance from the origin of the piece
                dists = [getDist(i, origin) for i in tile.spots]
                sortedPillars = [i for _, i in sorted(zip(dists, tile.spots))]
                sortedFilled = [i for _, i in sorted(zip(dists, tile.filled))]
                pillarLast = False
                for (binary, pillar) in zip(sortedFilled, sortedPillars):
                    if binary:
                        newPillar = sim.simxCopyPasteObjects(clientID, [pillarHandle], sim.simx_opmode_blocking)[1][0]
                        sim.simxSetObjectOrientation(clientID, pillarHandle, -1, [0, 0, 0], sim.simx_opmode_blocking)
                        pillarHandles.append(newPillar)
                        # make pillar, place in position
                        location = np.array([x for x in pillar])
                        location = np.concatenate((location/1000, [0.01]))*5
                        pillarLocations.append(location)
                        parentTiles.append(newTile)
                        # print(location)
                        sim.simxSetObjectPosition(clientID, newPillar, sim_handle_parent, location, sim.simx_opmode_blocking)
                        pillarLast = True
                    # time.sleep(0.8)

        # print("Out with the old!")
        if addons != None:
            for i in addons:
                if i[0] == 0:  # pillar
                    # add new pillar
                    # for pillar, place on most recently used tile
                    newPillar = sim.simxCopyPasteObjects(clientID, [pillarHandle], sim.simx_opmode_blocking)[1][0]
                    sim.simxSetObjectOrientation(clientID, newPillar, -1, [0, 0, 0], sim.simx_opmode_blocking)
                    location = np.array(i[1])  # convert to numpy, z component not included
                    location = np.concatenate((location / 1000, [0.01])) * 5
                    # print(location)
                    pillarHandles.append(newPillar)
                    parentTiles.append(newTile)
                    pillarLocations.append(location)
                    # uses local coords, just add to most recent tile
                    sim.simxSetObjectPosition(clientID, newPillar, sim_handle_parent, location, sim.simx_opmode_blocking)
                    time.sleep(0.3)
                    pillarLast = True # check whether a pillar was placed last
                elif i[0] == 1:  # tile
                    tile = i[1]
                    newestTile = tile
                    handle = sim.simxGetObjectHandle(clientID, tile.id, sim.simx_opmode_blocking)[1]
                    sim_handle_parent = handle
                    newTile = sim.simxCopyPasteObjects(clientID, [handle], sim.simx_opmode_blocking)[1][0]
                    # position
                    # print(tile.id)
                    origin = [x / 1000 for x in tile.origin]  # convert from mm to m
                    # print(origin)
                    origin = [i * 5 for i in origin]  # multiply by 5 because of coppelia scaling
                    origin[2] *= 0.9  # lower it a bit
                    # set rotation first
                    ori = sim.simxGetObjectQuaternion(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
                    oriQ = np.quaternion(ori[3], ori[0], ori[1], ori[2])
                    rot = tile.rotation
                    newQ = np.quaternion(np.cos(rot / 2), 0, 0, np.sin(rot / 2))
                    quat = newQ * oriQ
                    quat2 = [quat.x, quat.y, quat.z, quat.w]
                    sim.simxSetObjectQuaternion(clientID, newTile, -1, quat2, sim.simx_opmode_blocking)
                    sim.simxSetObjectPosition(clientID, newTile, -1, origin, sim.simx_opmode_blocking)
                    pillarLast = False
                # time.sleep(1)
    else:
        print ('Failed connecting to remote API server')

    # time.sleep(2)
    # figure out top tile's position, check if it's correct
    actualPos = sim.simxGetObjectPosition(clientID, newTile, -1, sim.simx_opmode_blocking) # should just be last one called
    initialPos = origin
    # can guarantee position will be less than 0.05 if tower collapses
    thresh = 0.55

    collapsed = 0
    # check distances of pillars

    time.sleep(0.1)
    count = 0
    # print(pillarHandles)
    # print(pillarLocations)
    # print("Distances:")

    # only check for last floor
    for i in pillarHandles:
        sim_handle_parent = parentTiles[count]
        if sim_handle_parent == newTile: # should be last one
            pos = sim.simxGetObjectPosition(clientID, i, sim_handle_parent, sim.simx_opmode_blocking)
            # print(pos)
            # print(pillarLocations[count])
            time.sleep(0.3)
            dist = np.sqrt((pillarLocations[count][0] - pos[1][0]) ** 2 + (pillarLocations[count][1] - pos[1][1]) ** 2)
            # print(dist)
            # print()
            if dist > 0.1:
                collapsed = 1
                sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
                sim.simxFinish(clientID)
                return collapsed
        count += 1

    pos = sim.simxGetObjectOrientation(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
    # # check angle of top tile. If it has pillar on it and angle is bad, collapsed. If no pillars then fine
    if np.abs(pos[0]) + np.abs(pos[1]) > 0.1:
        collapsed = 1
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        return collapsed


    if actualPos[1][2] / initialPos[2] <= thresh and len(tower) != 1:
        collapsed = 1
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        return collapsed
        # print(actualPos[1][2] / initialPos[2])


    # Now close the connection to CoppeliaSim:
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)
    # print('Program ended')
    return collapsed