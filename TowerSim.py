# generates towers for low level agent, saves to file

import random
import numpy as np
import sys
import pickle
import SawyerSim

printstuff = False
rad = 10

# spots have diameter = 20 mm, rad = 10 mm
# pillars have diameter = 12 mm, rad = 6 mm
# pillars have height of 45 mm

# given a coordinate according to local origin, world origin (where (0,0) is in world coords) and rotation matrix,
# transform coordinate from local system to world
# rotate should be 2x2 since working in 2D
def tupleTransform(t1, origin, rotate):
    if len(t1) != len(origin):
        print("Tupples are not the same size")
        return ()
    t1 = np.array(t1)
    origin = np.array(origin)
    newT = np.matmul(t1, rotate) + origin
    return newT

# check if these spots are too close (2 radii)
def checkCollide(t1, t2):
    x = t2[0] - t1[0]
    y = t2[1] - t1[1]
    dist = np.sqrt(x**2 + y**2)
    if dist >= 2*rad:  # no collision
        return False
    else: # collision
        return True

def getRotateMat(angle):
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotate

# gets distance between p1 and p2
def getDist(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    dist = np.sqrt(x**2 + y**2)
    return dist

# given a point in a list of clusters, finds nearst
def closestCluster(point, clusters):
    min = sys.maxsize
    count = 0
    for k in clusters:
        dist = getDist(point, k)
        if dist < min:
            min = dist
            cluster = count
        count += 1
    return cluster


# takes in list of coordinates, binary list representing which are filled, and the number of clusters to find, finds most balanced set
def getClusters(spots, filled, numClusters):
    alpha = 0.5

    num = len(spots)
    pillars = []
    for k in range(num):
        if filled[k] == 1:
            pillars.append(spots[k])

    maxX = int(max(pillars, key=lambda x:x[0])[0])
    maxY = int(max(pillars, key=lambda x:x[1])[1])
    minX = int(min(pillars, key=lambda x:x[0])[0])
    minY = int(min(pillars, key=lambda x:x[1])[1])

    if numClusters > len(range(minX, maxX)):  # sometimes all spots have the same x or y coord
        if printstuff:
            print("XRange too small for clusters, defaulting to minX")
        xs = [minX] * numClusters
    else:
        xs = random.sample(range(minX, maxX), numClusters)

    if numClusters > len(range(minY, maxY)):
        if printstuff:
            print("YRange too small for clusters, defaulting to minY")
        ys = [minY] * numClusters
    # randomly place numCluster points within the bounds of min/max
    else:
        ys = random.sample(range(minY, maxY), numClusters)

    origin = []
    for k in range(numClusters):
        origin.append([xs[k], ys[k]])

    thresh = 300
    error = thresh + 1
    count = 0
    # iteratively for each pillar, pull the cluster center closest to the pillar closer
    while error > thresh and count < 100:
        error = 0
        for k in pillars:
            nearest = closestCluster(k, origin) # index of origin that is closest to the point
            xDist = k[0] - origin[nearest][0]  # distance between
            yDist = k[1] - origin[nearest][1]
            # move proportionally to distance
            origin[nearest][0] += alpha*xDist
            origin[nearest][1] += alpha*yDist

            error += np.sqrt(alpha*xDist**2 + alpha*yDist**2)
        count += 1
    if printstuff == True:
        print("Cluster error: " + str(error))
    return origin

class Tile:
    # spots is array of tupples, contatining distance from com of tile in x and y (mm)
    # rotation is angle of rotation the tile is in (radians)
    def __init__(self, spots, rotation=0, origin=(0,0)):
        self.spots = spots
        self.rotation = rotation
        self.origin = origin

# list of all tiles and their spots relative to their origin (mm)
# only ints allowed
# Colours are red (R), yellow (Y), black (K), blue (B), and white (W)
# colours aren't necessary for low level agent, so not included with info
# Dagger 01
tile01 = Tile([(-3, 47), (20, 10), (-1, -42)]) # (R, Y, R)
# Majora's Mask
tile02 = Tile([(27, 10), (-5, -26), (-22, 17)]) # (K, K, R)
# Shard
tile03 = Tile([(-61, -15), (-5, 13), (16, -18), (55, -3)]) # (R, Y, R, W)
# Square Spiral
tile04 = Tile([(-34, 45), (-11, -10), (13, 10), (35, -45)]) # (Y, K, K, Y)
# Toucan
tile05 = Tile([(-34, -11), (-6, -22), (23, 25), (34, -11)]) # (R, K, Y, Y)
# Crescent
tile06 = Tile([(-31, 24), (2, 38), (34, 21)]) # (Y, R, W)
# Half Ax Head
tile07 = Tile([(-28, -25), (2, 34), (9, -20)]) # (Y, Y, Y)
# Ax
tile08 = Tile([(-55, 36), (-30, -7), (-15, 48), (83, -3)]) # (Y, K, K, Y)
# Triskelion
tile09 = Tile([(-53, -30), (-1, 61), (53, -30)]) # (K, Y, W)
# Orchid
tile10 = Tile([(-33, -34), (-13, 38), (-3, -35), (31, -19), (36, 39)]) # (Y, K, Y, B, K)
# Red Square
tile11 = Tile([(-24, 28), (-13, -28), (28, -29), (32, 31)]) # (R, R, R, R)
# Massive
tile12 = Tile([(-65, 0), (-39, 0), (1, -39), (1, 39), (41, 0)]) # (W, Y, R, B, Y)
# Square
tile13 = Tile([(-11, -12), (-11, 11), (11, 11), (41, 41), (42, -42)]) # (R, W, R, W, Y)
# Ax Head
tile14 = Tile([(-15, 37), (0, 0), (14, 38)]) # (Y, R, B)
# Flame
tile15 = Tile([(-23, 12), (-1, 29), (29, 13)]) # (Y, R, Y)
# Shuriken
tile16 = Tile([(-24, -11), (0, 0), (1, 26), (22, -16)]) # (Y, K, R, Y)
# Stairs
tile17 = Tile([(-70, 2), (-22, 28), (16, 3), (70, 4)]) # (Y, Y, R, Y)
# Spaceship
tile18 = Tile([(-26, 32), (0, 9), (26, 32)]) # (R, K, Y)

tiles = [tile01, tile02, tile03, tile04, tile05, tile06, tile07, tile08, tile09, tile10, tile11, tile12, tile13, tile14, tile15, tile16, tile17, tile18]

# takes in array of coordinates and binary array indicating whether they are occupied (1) or not (0)
# returns all spots that are occupied
def getOccupied(spots, filled):
    occupied = []
    count1 = 0
    for j in filled: # goes up to 3
        temp = []
        count2 = 0
        num = len(j)
        for i in range(num):
            if j[i] == 1:
                spotsTemp = spots[count1][count2]
                temp.append(spotsTemp)
            count2 += 1
        occupied.append(temp)
        count1 += 1
    if printstuff:
        print("Occupied:")
        print(occupied)
    return occupied



# builds random tower
# returns occupied spots and origin of tiles
# can make plane based on that
def build():
    allWorldSpots = []
    allFilled = []
    # randomly choose 3 tiles and place in fixed origin spots
    firstFloorId = random.sample(range(0, 18), 3)  # corresponds to index in tiles, so number 0 = tile01
    firstFloors = [tiles[firstFloorId[0]], tiles[firstFloorId[1]], tiles[firstFloorId[2]]]
    if printstuff == True:
        print(firstFloorId)
    origin1 = [(200, 280), (300, 280), (370, 240)]  # mm from origin of world coord
    count = 0
    for i in firstFloors:
        i.origin = origin1[count]
        count += 1

    origins = [origin1]


    # randomly choose level (1, 2, 3)
    levels = random.sample(range(1, 4), 1)[0]  # either 1 2 or 3
    if printstuff == True:
        print("Going up to level " + str(levels))
    filledSpots = []

    # always going to have 1st level at least
    # choose configuration such that pillar spots aren't on top of each other
    # figure out spots - first can stay where it is, adjust 2nd and 3rd around it
    # assuming no rotation, where would spots be in world coordinates
    firstSpotsWorld = []
    rotate = np.identity(2)
    angle = 0
    for spot in firstFloors[0].spots:
        firstSpotsWorld.append(tupleTransform(spot, origin1[0], rotate)) # switching to arrays instead of tuples
    # check for collisions between spots, if so, rotate all with rotation matrix. Repeat until no collisions
    # collisions = being less than 2 radii away from another point
    collide = True
    count = 0
    temp = firstFloors[1].spots  # get array of spots
    while collide:
        collide = False
        i = temp[count] # iterate through current tile in local coordinates
        newCoord = tupleTransform(i, origin1[1], rotate)
        for j in firstSpotsWorld:
            if checkCollide(j, newCoord) == True:
                collide = True
                count = 0 # reset and check against all coordinates again
                # adjust rotation matrix by 5 degrees
                angle += 0.0872665
                if printstuff == True:
                    print("Rotate 1.2! " + str(angle))
                rotate = getRotateMat(angle)
                if angle > 2 * np.pi:
                    print("no good position found for tile 1.2")
                    return -1, -1, -1
                break
            count += 1
    # made it all the way through without collision, set angle
    firstFloors[1].rotation = angle
    # add to world spots
    for spot in firstFloors[1].spots:
        firstSpotsWorld.append(tupleTransform(spot, origin1[1], rotate)) # just append, don't need to worry about distinguishing 1st and 2nd floor

    # on to the 3rd tile of 1st floor - pretty much same as 2nd tile
    collide = True
    count = 0
    rotate = np.identity(2)
    angle = 0
    temp = firstFloors[2].spots
    while collide:
        collide = False
        i = temp[count]
        newCoord = tupleTransform(i, origin1[2], rotate)
        for j in firstSpotsWorld:
            if checkCollide(j, newCoord) == True:
                collide = True
                count = 0
                angle += 0.0872665
                if printstuff == True:
                    print("Rotate 1.3! " + str(angle))
                rotate = getRotateMat(angle)
                if angle > 2 * np.pi:
                    print("no good position found for tile 1.3")
                    return -1, -1, -1
                break
            count += 1

    firstFloors[2].rotation = angle
    for spot in firstFloors[2].spots:
        firstSpotsWorld.append(tupleTransform(spot, origin1[1], rotate))
    allWorldSpots.append(firstSpotsWorld)

    if printstuff == True:
        print("Spots level 1: " + str(firstSpotsWorld))
    # randomly fill 1st level with pillars (dependent on level chosen -> higher = more)
    # for level 1: 0 - 100%, level 2: 50-100%, level 3: 80-100%
    numPillars1 = len(firstSpotsWorld)
    filled1 = np.zeros(numPillars1)  # binary, either filled spot or not

    # here 1 refers to the level number, not the tile number
    if levels == 1:
        # pick number between 0 and the number of pillars
        numChosen1 = random.sample(range(0, numPillars1), 1)[0]
        chosen1 = random.sample(range(0, numPillars1), numChosen1)
        # set chosen pillars to 1 (filled)
        for i in chosen1:
            filled1[i] = 1

    elif levels == 2:
        numChosen1 = random.sample(range(numPillars1//2 - 1, numPillars1), 1)[0] # limit the number of pillars to be between half and all
        chosen1 = random.sample(range(0, numPillars1), numChosen1)
        for i in chosen1:
            filled1[i] = 1

    elif levels == 3:
        numChosen1 = random.sample(range(int(numPillars1 * 0.8)-1, numPillars1), 1)[0]  # limit the number of pillars to be between 80% and all
        chosen1 = random.sample(range(0, numPillars1), numChosen1)
        for i in chosen1:
            filled1[i] = 1

    # requires 2D array
    occupied1 = getOccupied([firstSpotsWorld], [filled1])
    filledSpots.append(occupied1)
    allFilled.append(filled1)

    # now build second level (if applicable)
    if levels > 1:
        # for second floor either 2 or 3 tiles
        numTiles2 = random.sample(range(2, 4), 1)[0]
        secondFloorId = random.sample(range(0, 18), numTiles2)
        if printstuff == True:
            print(secondFloorId)
        secondFloors = []
        for i in secondFloorId:
            secondFloors.append(tiles[i])

        origin2 = getClusters(firstSpotsWorld, filled1, numTiles2)

        count = 0
        for i in secondFloors:
            i.origin = origin2[count]
            count += 1
        origins.append(origin2)
        if printstuff == True:
            print("Level 2 origins: " + str(origin2))
        # orient so no collisions
        secondSpotsWorld = []
        # place 1st floor of 2nd level in neutral
        for spot in secondFloors[0].spots:
            secondSpotsWorld.append(tupleTransform(spot, origin2[0], rotate))

        collide = True
        count = 0
        temp = secondFloors[1].spots
        rotate = np.identity(2)
        angle = 0
        while collide:
            collide = False
            i = temp[count]  # iterate through current tile in local coordinates
            newCoord = tupleTransform(i, origin2[1], rotate)
            for j in secondSpotsWorld:
                if checkCollide(j, newCoord) == True:
                    collide = True
                    count = 0
                    angle += 0.0872665
                    if printstuff == True:
                        print("Rotate 2.2! " + str(angle))
                    rotate = getRotateMat(angle)
                    if angle > 2 * np.pi:
                        print("no good position found for tile 2.2")
                        return -1, -1, -1
                    break
                count += 1
        secondFloors[1].rotation = angle
        # add to world spots
        for spot in secondFloors[1].spots:
            secondSpotsWorld.append(tupleTransform(spot, origin2[1], rotate))

        if numTiles2 == 3: # must add 3rd tile
            collide = True
            count = 0
            temp = secondFloors[2].spots
            rotate = np.identity(2)
            angle = 0
            while collide:
                collide = False
                i = temp[count]
                newCoord = tupleTransform(i, origin2[2], rotate)
                for j in secondSpotsWorld:
                    if checkCollide(j, newCoord) == True:
                        collide = True
                        count = 0
                        angle += 0.0872665
                        if printstuff == True:
                            print("Rotate 2.2! " + str(angle))
                        rotate = getRotateMat(angle)
                        if angle > 2 * np.pi:
                            print("no good position found for tile 2.3")
                            return -1, -1, -1
                        break
                    count += 1
            secondFloors[2].rotation = angle
            for spot in secondFloors[2].spots:
                secondSpotsWorld.append(tupleTransform(spot, origin2[2], rotate))

        allWorldSpots.append(secondSpotsWorld)
        if printstuff == True:
            print("Spots level 2: " + str(secondSpotsWorld))

        # fill in spots
        numPillars2 = len(secondSpotsWorld)
        filled2 = np.zeros(numPillars2)

        if levels == 2: # 0 to 100%
            numChosen2 = random.sample(range(0, numPillars2), 1)[0]
            chosen2 = random.sample(range(0, numPillars2), numChosen2)
            for i in chosen2:
                filled2[i] = 1

        elif levels == 3: # 60% to 100%
            numChosen2 = random.sample(range(int(numPillars2 * 0.6) - 1, numPillars2), 1)[0]
            chosen2 = random.sample(range(0, numPillars2), numChosen2)
            for i in chosen2:
                filled2[i] = 1

        occupied2 = getOccupied([secondSpotsWorld], [filled2])
        filledSpots.append(occupied2)
        allFilled.append(filled2)
    
    # same as prev
    if levels > 2:
        # for third floor either 1 or 2 tiles
        numTiles3 = random.sample(range(1, 3), 1)[0]
        thirdFloorId = random.sample(range(0, 18), numTiles3)
        if printstuff == True:
            print(thirdFloorId)
        thirdFloors = []
        for i in thirdFloorId:
            thirdFloors.append(tiles[i])

        origin3 = getClusters(secondSpotsWorld, filled2, numTiles3)
        count = 0
        for i in firstFloors:
            i.origin = origin1[count]
            count += 1
        origins.append(origin3)
        if printstuff == True:
            print("Level 3 origins: " + str(origin3))
        # orient so no collisions
        thirdSpotsWorld = []
        rotate = np.identity(2)
        # place 1st floor of 3rd level in neutral
        for spot in thirdFloors[0].spots:
            thirdSpotsWorld.append(tupleTransform(spot, origin3[0], rotate))

        if numTiles3 == 2:
            collide = True
            count = 0
            temp = thirdFloors[1].spots
            angle = 0
            while collide:
                collide = False
                i = temp[count]
                newCoord = tupleTransform(i, origin3[1], rotate)
                for j in thirdSpotsWorld:
                    if checkCollide(j, newCoord) == True:
                        collide = True
                        count = 0
                        angle += 0.0872665
                        if printstuff == True:
                            print("Rotate 3.2! " + str(angle))
                        rotate = getRotateMat(angle)
                        if angle > 2 * np.pi:
                            print("no good position found for tile 3.2")
                            return -1, -1, -1
                        break
                    count += 1
            # made it all the way through without collision, set angle
            thirdFloors[1].rotation = angle
            # add to world spots
            for spot in thirdFloors[1].spots:
                thirdSpotsWorld.append(tupleTransform(spot, origin3[1], rotate))
        allWorldSpots.append(thirdSpotsWorld)
        if printstuff == True:
            print("Spots level 3: " + str(thirdSpotsWorld))

        # randomly fill from 0 to 100%
        numPillars3 = len(thirdSpotsWorld)
        filled3 = np.zeros(numPillars3)
        # need at least 1 free
        numChosen3 = random.sample(range(0, numPillars3), 1)[0]
        chosen3 = random.sample(range(0, numPillars3), numChosen3)
        for i in chosen3:
            filled3[i] = 1

        occupied3 = getOccupied([thirdSpotsWorld], [filled3])
        filledSpots.append(occupied3)
        allFilled.append(filled3)


    if printstuff == True:
        print("Origins:")
        print(origins)
        print("All Filled:")
        print(allFilled)
        print("All spots:")
        print(allWorldSpots)
        print("Occupied Final:")
        getOccupied(allWorldSpots, allFilled)


    return allWorldSpots, allFilled, origins


if __name__ == "__main__":
    numTowers = 50
    fileName = "TowerModels" + str(numTowers) + ".txt"
    file = open(fileName, 'wb')
    for _ in range(numTowers):
        goal = [-1]
        spots = -1
        while goal[0] == -1 or spots == -1:
            spots, filled, origins = build()
            if spots == -1:
                continue # won't be able to get goal
            goal = SawyerSim.getGoal(spots, filled)
        pickle.dump([spots, filled, origins, goal], file)
    file.close()

