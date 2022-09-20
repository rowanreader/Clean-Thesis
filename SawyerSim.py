# given initial state (forward kinematics), and goal destination, moves Sawyer robot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import TowerSim as tower
import random
import tensorflow as tf
import sys
import pickle
import math


# given joints and end effector position, plot (also plot goal point if it's given)
def plot(joints, endEffector, goal=None):
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot3D(joints[:,0], joints[:,1], joints[:,2], '-r')
    ax.plot3D(endEffector[:,0], endEffector[:,1], endEffector[:,2], '.-b')
    ax.plot3D(goal[0], goal[1], goal[2], '*g')
    plt.show()

def DH(a, alpha, d, th):
    # T = [[tf.cast(tf.cos(th), tf.float32),  -tf.cast(tf.sin(th), tf.float32)*tf.cast(tf.cos(alpha), tf.float32), tf.cast(tf.sin(th), tf.float32)*tf.cast(tf.sin(alpha), tf.float32),  tf.cast(a, tf.float32)*tf.cast(tf.cos(th), tf.float32)],
    #      [tf.cast(tf.sin(th), tf.float32),  tf.cast(tf.cos(th), tf.float32)*tf.cast(tf.cos(alpha), tf.float32),  -tf.cast(tf.cos(th), tf.float32)*tf.cast(tf.sin(alpha), tf.float32), tf.cast(a, tf.float32)*tf.cast(tf.sin(th), tf.float32)],
    #      [0,                tf.cast(tf.cast(tf.sin(alpha), tf.float32), tf.float32),                tf.cast(tf.cast(tf.cos(alpha), tf.float32), tf.float32),                d],
    #      [0,                    0,                              0,                      1]]
    T = [[np.cos(th),  -np.sin(th)*np.cos(alpha), np.sin(th)*np.sin(alpha),  a*np.cos(th)],
         [np.sin(th),  np.cos(th)*np.cos(alpha),  -np.cos(th)*np.sin(alpha), a*np.sin(th)],
         [0,                np.sin(alpha),                np.cos(alpha),                d],
         [0,                    0,                              0,                      1]]

    return T

# given angles of the Sawyer robot, finds end effector position
def FK(angles):
    j0 = [0,0,0]
    d = [317, 192.5, 400, 168.5, 400, 136.3, 133.75] # link length
    a = [81, 0, 0, 0, 0, 0, 0]
    alpha = [-math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
    th = [0, 3*math.pi/2, 0, math.pi, 0, math.pi, 3*math.pi/2] # added to angles
    # get all T matrices
    T1 = DH(a[0], alpha[0], d[0], th[0]+ angles[0])
    T2 = DH(a[1], alpha[1], d[1], th[1]+ angles[1])
    T3 = DH(a[2], alpha[2], d[2], th[2]+ angles[2])
    T4 = DH(a[3], alpha[3], d[3], th[3]+ angles[3])
    T5 = DH(a[4], alpha[4], d[4], th[4]+ angles[4])
    T6 = DH(a[5], alpha[5], d[5], th[5]+ angles[5])
    TP = DH(a[6], alpha[6], d[6], th[6]+ angles[6])

    id = [0, 0, 0, 1]# extracts last col
    j1 = np.matmul(T1, id) # will be a 1D array
    j1 = j1[0:3] # just 1st 3 nums

    j1_ = np.matmul(T1, [-81, 0, 0, 1])
    j1_ = j1_[0:3]

    T12 = np.matmul(T1, T2)
    j2 = np.matmul(T12, id)
    j2 = j2[0:3]

    T13 = np.matmul(T12, T3)
    j3 = np.matmul(T13, id)
    j3 = j3[0:3]

    T14 = np.matmul(T13, T4)
    j4 = np.matmul(T14, id)
    j4 = j4[0:3]

    T15 = np.matmul(T14, T5)
    j5 = np.matmul(T15, id)
    j5 = j5[0:3]

    T16 = np.matmul(T15, T6)
    j6 = np.matmul(T16, id)
    j6 = j6[0:3]

    # technically this is the end effecctor and a joint
    T1P = np.matmul(T16, TP)
    jP = np.matmul(T1P, id)
    jP = jP[0:3]

    joints = np.array([j0, j1_, j1, j2, j3, j4, j5, j6, jP])

    claw = 40 # size of claw lengths in mm (all identical)

    A = [claw, 0, 0, 1]
    B = [claw, 0, claw, 1]
    C = [-claw, 0, 0, 1]
    D = [-claw, 0, claw, 1]

    A = np.matmul(T1P, A)
    B = np.matmul(T1P, B)
    C = np.matmul(T1P, C)
    D = np.matmul(T1P, D)

    actuators = np.array([B[0:3], A[0:3], C[0:3], D[0:3]])
    return jP, joints, actuators


# given angles, calculates Jacobian
def Jacobian(angles):
    th1 = angles[0]
    th2 = angles[1]
    th3 = angles[2]
    th4 = angles[3]
    th5 = angles[4]
    th6 = angles[5]
    l1 = 3170
    l2 = 1925
    l3 = 4000
    l4 = 1685
    l5 = 4000
    l6 = 1363
    l7 = 1337.5
    jacob = [[l4*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) - l5*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) - 81*np.sin(th1) + l6*(np.sin(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*(np.cos(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) + np.sin(th6)*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)))) - l3*np.cos(th2)*np.sin(th1), l6*(np.sin(th5)*(np.cos(th1)*np.sin(th2)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th3)*np.cos(th4)) + np.cos(th1)*np.cos(th2)*np.cos(th5)*np.sin(th3)) - l7*(np.cos(th6)*(np.cos(th1)*np.cos(th4)*np.sin(th2) - np.cos(th1)*np.cos(th2)*np.cos(th3)*np.sin(th4)) + np.sin(th6)*(np.cos(th5)*(np.cos(th1)*np.sin(th2)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th3)*np.cos(th4)) - np.cos(th1)*np.cos(th2)*np.sin(th3)*np.sin(th5))) - l5*(np.cos(th1)*np.cos(th4)*np.sin(th2) - np.cos(th1)*np.cos(th2)*np.cos(th3)*np.sin(th4)) - l3*np.cos(th1)*np.sin(th2) - l4*np.cos(th1)*np.cos(th2)*np.sin(th3), l6*(np.cos(th5)*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) + np.cos(th4)*np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) - l4*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) + l7*(np.sin(th6)*(np.sin(th5)*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) - np.cos(th4)*np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + np.cos(th6)*np.sin(th4)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l5*np.sin(th4)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)),   l7*(np.cos(th6)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.cos(th5)*np.sin(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4))) + l5*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - l6*np.sin(th5)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)),   l6*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l7*np.sin(th6)*(np.sin(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))), -l7*(np.sin(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) + np.cos(th6)*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)))), 0],    [81*np.cos(th1) + l5*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) + l4*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)) + l6*(np.sin(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l7*(np.cos(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) - np.sin(th6)*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)))) + l3*np.cos(th1)*np.cos(th2),                                             -np.sin(th1)*(l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + l5*np.cos(th4)*np.sin(th2) - l5*np.cos(th2)*np.cos(th3)*np.sin(th4) - l6*np.cos(th2)*np.cos(th5)*np.sin(th3) + l7*np.cos(th4)*np.cos(th6)*np.sin(th2) - l6*np.sin(th2)*np.sin(th4)*np.sin(th5) - l6*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.sin(th5) - l7*np.cos(th2)*np.cos(th3)*np.cos(th6)*np.sin(th4) - l7*np.cos(th2)*np.sin(th3)*np.sin(th5)*np.sin(th6) + l7*np.cos(th5)*np.sin(th2)*np.sin(th4)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.sin(th6)), l4*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) - l6*(np.cos(th5)*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) + np.cos(th4)*np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*(np.sin(th6)*(np.sin(th5)*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) - np.cos(th4)*np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) + np.cos(th6)*np.sin(th4)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l5*np.sin(th4)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)), - l5*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - l7*(np.cos(th6)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*np.sin(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4))) - l6*np.sin(th5)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)), - l6*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*np.sin(th6)*(np.sin(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))), -l7*(np.sin(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) - np.cos(th6)*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)))), 0],     [0, l4*np.sin(th2)*np.sin(th3) - l5*np.cos(th2)*np.cos(th4) - l3*np.cos(th2) - l7*np.cos(th2)*np.cos(th4)*np.cos(th6) - l5*np.cos(th3)*np.sin(th2)*np.sin(th4) - l6*np.cos(th5)*np.sin(th2)*np.sin(th3) + l6*np.cos(th2)*np.sin(th4)*np.sin(th5) - l6*np.cos(th3)*np.cos(th4)*np.sin(th2)*np.sin(th5) - l7*np.cos(th3)*np.cos(th6)*np.sin(th2)*np.sin(th4) - l7*np.cos(th2)*np.cos(th5)*np.sin(th4)*np.sin(th6) - l7*np.sin(th2)*np.sin(th3)*np.sin(th5)*np.sin(th6) + l7*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.sin(th2)*np.sin(th6), -np.cos(th2)*(l4*np.cos(th3) - l6*np.cos(th3)*np.cos(th5) + l5*np.sin(th3)*np.sin(th4) + l6*np.cos(th4)*np.sin(th3)*np.sin(th5) + l7*np.cos(th6)*np.sin(th3)*np.sin(th4) - l7*np.cos(th3)*np.sin(th5)*np.sin(th6) - l7*np.cos(th4)*np.cos(th5)*np.sin(th3)*np.sin(th6)),                                                                                                                                                      l5*np.sin(th2)*np.sin(th4) + l5*np.cos(th2)*np.cos(th3)*np.cos(th4) + l6*np.cos(th4)*np.sin(th2)*np.sin(th5) + l7*np.cos(th6)*np.sin(th2)*np.sin(th4) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th6) - l6*np.cos(th2)*np.cos(th3)*np.sin(th4)*np.sin(th5) - l7*np.cos(th4)*np.cos(th5)*np.sin(th2)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th5)*np.sin(th4)*np.sin(th6), l6*np.cos(th5)*np.sin(th2)*np.sin(th4) - l6*np.cos(th2)*np.sin(th3)*np.sin(th5) + l6*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5) + l7*np.cos(th2)*np.cos(th5)*np.sin(th3)*np.sin(th6) + l7*np.sin(th2)*np.sin(th4)*np.sin(th5)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.sin(th5)*np.sin(th6), l7*np.cos(th4)*np.sin(th2)*np.sin(th6) - l7*np.cos(th2)*np.cos(th3)*np.sin(th4)*np.sin(th6) + l7*np.cos(th2)*np.cos(th6)*np.sin(th3)*np.sin(th5) - l7*np.cos(th5)*np.cos(th6)*np.sin(th2)*np.sin(th4) - l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.cos(th6), 0]]

    return jacob

# given pillar base coord, center and radius to make sphere, check if sphere collides with cylinder (pillar)

def pillarCollision(pillar, center, radius):
    pRad = 6 # pillar radius
    # translate everything so that sphere is centered at origin
    newPillar = np.array(pillar) # need to avoid aliasing
    newPillar[2] = getHeight(pillar[2]) # convert to mm
    temp = newPillar - center
    # find out how close axis of pillar is to 'origin''
    dist = np.sqrt(temp[0]**2 + temp[1]**2 + temp[2]**2)
    if dist >= radius + pRad: # check that combined radii are less than dist between axes
        return False # no collision
    if printMe:
        print(newPillar, center)
    return True


# given plane (normal to z, so just z), x and y limits, center and radius of sphere, find if they intersect
def planeCollision(z, xlim, ylim, center, radius):
    # find distance from sphere center to plane height
    dist = center[2] - z # only need to worry about z coordinates because plane's normal is [0 0 1]
    # check that distance is less than radius of sphere. If it is more, no collision
    if dist >= radius:
        return False
    # now find point on plane that is closest
    # p = [center[0], center[1], z] # logically it would be this
    # check if it is within limits
    pmin = [center[0] - radius, center[1] - radius]
    pmax = [center[0] + radius, center[1] + radius]
    # check within z lim
    zlim = [z-radius, z+radius]
    if pmin[0] >= xlim[0] and pmax[0] <= xlim[1] and pmin[1] >= ylim[0] and pmax[1] <= ylim[1] and zlim[0] <= center[2] <= zlim[1]:
        return True
    return False

# given an array of origins, find min and max x values, add 5 cm to all (50 mm)
def getLimits(centers):
    xmin = centers[0][0]
    xmax = centers[0][0]
    ymin = centers[0][1]
    ymax = centers[0][1]
    offset = 50
    for i in centers:
        if i[0] > xmax:
            xmax = i[0] + offset
        elif i[0] < xmin: # should only ever be one or the other
            xmin = i[0] - offset

        if i[1] > ymax:
            ymax = i[1] + offset
        elif i[1] < ymin:
            ymin = i[1] - offset

    return (xmin, xmax), (ymin, ymax)

# checks whether the wrist or pillar in hand has collided with standing pillars or tiles
# spots is only occupied spots
# spots have diameter = 20 mm, rad = 10 mm
# pillars have diameter = 12 mm, rad = 6 mm
# pillars have height of 45 mm

def checkCollision(p, spots, origins):
    # check if p is clear within set rad
    wristRad = 30  # in mm, so 3 cm
    # wrist represented by sphere around p with rad 30
    # each pillar/spot is a coordinate in 3D (height based on level). It extends a certain height up and has rad 10
    for k in spots: # 3 levels
        for i in k:
            if pillarCollision(i, p, wristRad): # if true, pillar collision occurred
                return True, i

    level = 0
    for i in origins:
        # must find limits of x and y based on plane
        xlim, ylim = getLimits(i)
        height = getHeight(level)
        if planeCollision(height, xlim, ylim, p, wristRad): # if true, tile collision occurred
            return True, i
        level += 1

    return False, None  # if made to this point, no collision


# pick a spot on the top level that is empty
def getGoal(spots, filled):
    height = 45 # pillar height
    tileHeight = 2 # tile height
    levels = len(spots)-1
    empty = []
    count = 0
    # get all empty spots on top floor
    for i in spots[levels]:
        if filled[levels][count] == 0:
            temp = np.append(i, (levels+1)*height + tileHeight*(levels+1) + 30)  # gotta add z component, account for wrist rad
            empty.append(temp)

    if empty == []:
        if printMe:
            print("Oops, no spots available")
        goal = [-1]
        return goal

    # randomly choose one
    goal = random.choice(empty)
    return goal

# levels starts at 0
def getHeight(levels):
    height = 45  # pillars in mm
    tileHeight = 2 # thickness of cardboard
    return (levels * height + tileHeight * (levels + 1))


def get3D(occupied):
    count = 0
    occ3D = []
    for i in occupied: # should be 3 items
        height = getHeight(count)
        count += 1
        temp = []
        for j in i:
            temp.append(np.append(j,height))
        occ3D.append(temp)
    return occ3D


# takes in goal coord, current configuration, all spots, filled binary array, and origins of floors
# returns outcome (fail/success), joint angles, and end point location
def IK(goal, q, spots, filled, origins):
    occupiedTemp = tower.getOccupied(spots, filled)
    # must convert occupied into 3D data
    occupied = get3D(occupiedTemp)
    if plotMe:
        moviewriter = ani.PillowWriter()
        fig = plt.figure(1)
    p, joints, endEffector = FK(q) # p is current end joint, joints are angles the joints are at, end effector is for plotting only

    qOld = [0, 0, 0, 0, 0, 0, 0] # default configuration, can change to reflect actual default configuration
    count = 0
    threshold = 2000
    alpha = 0.8
    if plotMe:
        with moviewriter.saving(fig, 'IKsimulation.gif', dpi=100):
            while np.linalg.norm(np.subtract(qOld, q)) > 0.005 and count < threshold:
                count = count + 1

                diff = np.subtract(goal, p)
                jacob = Jacobian(q)
                dq = np.matmul(np.linalg.pinv(jacob), diff)
                qOld = q
                q = q + alpha * dq
                p, joints, endEffector = FK(q)

                plt.clf()
                ax = plt.axes(projection='3d')

                collide, pt = checkCollision(p, occupied, origins)
                # if collides, return that it collided
                if collide and printMe:
                    print("Collide!")
                    count = threshold # finish loop but then return
                    ax.plot3D(pt[0], pt[1], pt[2], '*m') # plot collision point

                ax.plot3D(p[0], p[1], p[2], '*c')
                ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], '.-r')
                ax.plot3D(endEffector[:, 0], endEffector[:, 1], endEffector[:, 2], '.-b')
                ax.plot3D(goal[0], goal[1], goal[2], '*g')

                moviewriter.grab_frame()
        moviewriter.finish()

    else:
        while np.linalg.norm(np.subtract(qOld, q)) > 0.005 and count < threshold:
            count = count + 1
            diff = np.subtract(goal, p)
            jacob = Jacobian(q)
            dq = np.matmul(np.linalg.pinv(jacob), diff)
            qOld = q
            q = q + alpha * dq
            p, joints, endEffector = FK(q)

            # check for collision
            collide, pt = checkCollision(p, occupied, origins)
            # if collides, return that it collided

            if collide:
                return -2, q, p  # collision, failed

    # if get here, success!
    if printMe:
        print(count)
    if count < threshold and printMe:
        print("Success!")
        print(p)
    if count == threshold:
        print("Exceeded thresh:")
        return -1, q, p
    return 1, q, p



printMe = False
plotMe = False

if __name__ == "__main__":
    # plotMe = True
    count = 0
    if plotMe:
        # for building tower from scratch:
        # spots, filled, origins = tower.build()  # all spots, binary array of occupied or not, origins
        # # based on spots, pick one of the unfilled ones from the top floor as a goal
        # goal = [-1]
        # while goal[0] == -1 and count < 100:
        #     goal = getGoal(spots, filled)
        #     count += 1
        # if count == 100:
        #     print("oops")
        #     sys.exit()

        # for reading tower in from file:
        num = 1000
        yAx = "Rewards"
        ver = str(num) + "_"
        fileName = "TowerModels" + str(num) + ".txt"
        f = open(fileName, 'rb')
        temp = pickle.load(f)
        f.close()
        spots = temp[0]
        filled = temp[1]
        origins = temp[2]
        goal = temp[3]
        occupied = tower.getOccupied(spots, filled)


        print("Goal " + str(goal))
        q = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # must be different from default configuration
        print(IK(goal, q, spots, filled, origins))

    else:
        while count < 10:
            # read in data
            spots, filled, origins = tower.build() # all spots, binary array of occupied or not, origins
            if spots == -1: # mistake in building, retry
                continue
            # based on spots, pick one of the unfilled ones from the top floor as a goal
            goal = getGoal(spots, filled)
            if goal[0] == -1: # no spot available
                continue

            q = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # start configuration
            print("Goal " + str(goal))
            IK(goal, q, spots, filled, origins)

            count += 1

