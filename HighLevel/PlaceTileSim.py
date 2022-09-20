# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

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

def simulate(tower):
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to CoppeliaSi
    if clientID != -1:
        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print ('Remote API function call returned with error code: ', res)
            sys.quit()
        sim_handle_parent = -1
        pillarHandle = sim.simxGetObjectHandle(clientID, "Pillar", sim.simx_opmode_blocking)[1]
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        for level in tower:
            for tile in level:
                # for each tile, place at spot
                # get handle for tile
                handle = sim.simxGetObjectHandle(clientID, tile.id, sim.simx_opmode_blocking)[1]
                newTile = sim.simxCopyPasteObjects(clientID, [handle], sim.simx_opmode_blocking)[1][0]
                # position
                origin = [x/1000 for x in tile.origin] # convert from mm to m
                origin = [i*5 for i in origin] # multiply by 5 because of coppelia scaling
                origin[2] *= 0.7 # lower it a bit
                # set rotation first
                ori = sim.simxGetObjectQuaternion(clientID, newTile, sim_handle_parent, sim.simx_opmode_blocking)[1]
                oriQ = np.quaternion(ori[3], ori[0], ori[1], ori[2])
                rot = tile.rotation
                newQ = np.quaternion(np.cos(rot / 2), 0, 0, np.sin(rot / 2))
                quat = newQ * oriQ
                quat2 = [quat.x, quat.y, quat.z, quat.w]
                sim.simxSetObjectQuaternion(clientID, newTile, sim_handle_parent, quat2, sim.simx_opmode_blocking)
                sim.simxSetObjectPosition(clientID, newTile, sim_handle_parent, origin, sim.simx_opmode_blocking)
                # rotation
                # for each tile, fill pillars
                sim_handle_parent = newTile
                # order pillars by distance from the origin of the piece
                dists = [getDist(i, origin) for i in tile.spots]
                sortedPillars = [i for _, i in sorted(zip(dists, tile.spots))]
                sortedFilled = [i for _, i in sorted(zip(dists, tile.filled))]
                for (binary, pillar) in zip(sortedFilled, sortedPillars):
                    if binary:
                        newPillar = sim.simxCopyPasteObjects(clientID, [pillarHandle], sim.simx_opmode_blocking)[1][0]
                        sim.simxSetObjectOrientation(clientID, pillarHandle, -1, [0, 0, 0], sim.simx_opmode_blocking)
                        # make pillar, place in position
                        location = np.array([x for x in pillar])
                        location = np.concatenate((location/1000, [0.01]))*5
                        sim.simxSetObjectPosition(clientID, newPillar, sim_handle_parent, location, sim.simx_opmode_blocking)

    else:
        print ('Failed connecting to remote API server')

    time.sleep(0.1)
    # figure out top tile's position, check if it's correct
    actualPos = sim.simxGetObjectPosition(clientID, newTile, -1, sim.simx_opmode_blocking) # should just be last one called
    initialPos = origin
    # can guarantee position will be less than 0.05 if tower collapses
    thresh = 0.55

    collapsed = 0
    pos = sim.simxGetObjectOrientation(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
    # check angle of top tile. If it has pillar on it and angle is bad, collapsed. If no pillars then fine
    if actualPos[1][2] < 0.1:
        collapsed = 1
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        return collapsed

    if np.abs(pos[0]) + np.abs(pos[1]) > 0.1:
        collapsed = 1
        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        return collapsed

    # Now close the connection to CoppeliaSim:
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)
    return collapsed