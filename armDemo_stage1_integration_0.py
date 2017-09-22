from acerlib.RemoteSession import findX11DisplayPort

findX11DisplayPort()
import importlib
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import array as npa
from scipy import interpolate

from VrepPythonApi import vrep
from VrepPythonApi import vrep_ext
from armDemo import ArmDemo, waitSecs, ProxSensor_top
from armDemo_stage1_loadData import imgPreprocess
from acerlib import logging_ext

from keras.models import load_model

# ============================================================================ #
#                                  parameters                                  #
# ============================================================================ #
spotlightRadius_m = 0.6  # in meter

# ============================================================================ #
#                               Mapping function                               #
# ============================================================================ #

xRange = (0, 222)
yRange = (0, 110)
xBound = (0, 1)
yBound = (0, 0.5)

m2p_x = interpolate.interp1d(xBound, xRange)
m2p_y = interpolate.interp1d(yBound, yRange)

p2m_x = interpolate.interp1d(xRange, xBound)
p2m_y = interpolate.interp1d(yRange, yBound)

# ============================================================================ #
#                                 Preprocessing                                #
# ============================================================================ #


# ============================================================================ #
#                                  Model init                                  #
# ============================================================================ #
# mPath = 'armDemo_stage1'
# mID = 'model_4_2'
# m = load_model(os.path.join(mPath, '{:s}_CModelCheckpoint_best.hdf5'.format(mID)))

mPath = "armDemo_stage1/model_history_model_4_2/model_4_2_CModelCheckpoint_0293.hdf5"
m = load_model(mPath)

# ============================================================================ #
#                               Connect to V-rep                               #
# ============================================================================ #
arm = ArmDemo('192.168.1.170')
prox = ProxSensor_top(arm.clientID)


# ============================================================================ #
#                             remove obj out of box                            #
# ============================================================================ #


def removeOutOfBoxObjs(clientID, h_objs, xBound, yBound):
    pos = arm.getObjPosition(h_objs)
    pos = npa(pos)

    # remove
    xf = np.logical_or(pos[:, 0] < xBound[0], pos[:, 0] > xBound[1])
    yf = np.logical_or(pos[:, 1] < yBound[0], pos[:, 1] > yBound[1])
    iOutOfBox = np.logical_or(xf, yf)
    if any(iOutOfBox):
        iiOutOfBox = h_objs[iOutOfBox]
        for i_h_obj in iiOutOfBox:
            vrep.simxRemoveObject(clientID, i_h_obj, vrep.simx_opmode_blocking)
        h_objs = np.delete(h_objs, np.where(iOutOfBox))

    return h_objs

# h_objs = removeOutOfBoxObjs(arm.clientID, h_objs, xBound, yBound)


# ============================================================================ #
#                             Detect highest point                             #
# ============================================================================ #
def detectHighestPoint(prox):
    p_highest = prox.getData()[0:2]
    for i in range(2):
        if p_highest[i] < 0:
            p_highest[i] = 0
    if p_highest[0] > xBound[1]:
        p_highest[0] = xBound[1]
    if p_highest[1] > yBound[1]:
        p_highest[1] = yBound[1]
    return p_highest


# p_highest = detectHighestPoint(prox)


# ============================================================================ #
#                                 Find centres                                 #
# ============================================================================ #
def findCentres(arm, m, isPlot=False):
    # Get visual data
    img = arm.getImageFromCam(isZeroToOne=True)
    img = imgPreprocess(img)
    img = np.expand_dims(img, 0)

    # create centre location image
    img_c = m.predict(img).squeeze()

    if isPlot:
        plt.ion()
        plt.imshow(img_c.squeeze())
        plt.show()
        plt.pause(0.1)

    return img_c


# img_c = findCentres(arm, m, True)


# ============================================================================ #
#                       Find highest centre in spotlight                       #
# ============================================================================ #
def findTargetCentre(img_c, p_highest, xRange, yRange, spotlightRadius_m):
    pix_highest = npa([m2p_x(p_highest[0]), m2p_y(p_highest[1])]).round()

    # create point list
    x, y = np.meshgrid(np.arange(xRange[1]), np.arange(yRange[1]))
    xy = zip(x.flatten(), y.flatten())
    xy = npa(list(xy))

    # compute point in range
    pix_dist = ((xy[:, 0] - pix_highest[0]) ** 2 + (xy[:, 1] - pix_highest[1]) ** 2) ** (1 / 2)
    spotlightRadius_pix = m2p_x(spotlightRadius_m)
    inRange = np.argwhere(pix_dist < spotlightRadius_pix)
    xy_inRange = xy[inRange.squeeze(), :]

    # find the highest point in range
    img_c_f = np.flipud(img_c)
    value_inRange = img_c_f[xy_inRange[:, 1], xy_inRange[:, 0]]  # become yx
    highC_inRange_i = np.argmax(value_inRange)
    highC_inRange = xy_inRange[highC_inRange_i, :]  # this is xy
    c_xy_m = npa([p2m_x(highC_inRange[0]), p2m_y(highC_inRange[1])])

    # sorted
    sortC_inRange_i = np.argsort(value_inRange)[::-1]
    cSort_xy_pix = xy_inRange[sortC_inRange_i, :]  # this is xy

    return c_xy_m, highC_inRange, cSort_xy_pix


# c_xy_m = findTargetCentre(img_c, p_highest, xRange, yRange, spotlightRadius_m)


# ============================================================================ #
#                                     Run!                                     #
# ============================================================================ #
nObj = 10
nRun = 5
xBound2 = (0.15, 0.85)
yBound2 = (0.1, 0.42)

for iRun in range(nRun):
    # restart simulation
    arm.restartSim()

    # create obj
    h_objs = npa(arm.createRabndomObj(nObj))
    waitSecs(1)
    h_objs = removeOutOfBoxObjs(arm.clientID, h_objs, xBound2, yBound2)

    while len(h_objs) > 0:
        p_highest = detectHighestPoint(prox)
        img_c = findCentres(arm, m, True)
        c_xy_m, highC_inRange, cSort_xy_pix = findTargetCentre(img_c, p_highest, xRange, yRange, spotlightRadius_m)

        # ============================================================================ #
        #                                   Move arm                                   #
        # ============================================================================ #
        # move above
        arm.moveTo([c_xy_m[0], c_xy_m[1], 0.25])
        arm.waitToDistination()

        # detect obj dist
        dist = arm.getProxSensorData()
        if dist == 0:
            for iSortC in range(cSort_xy_pix.shape[0]):
                cXY = cSort_xy_pix[iSortC, :]
                c_xy_m = npa([p2m_x(cXY[0]), p2m_y(cXY[1])])

                arm.moveTo([c_xy_m[0], c_xy_m[1], 0.25])
                arm.waitToDistination()
                dist = arm.getProxSensorData()

                if dist > 0:
                    break

        # move down
        arm.moveTo([c_xy_m[0], c_xy_m[1], 0.25 - dist + 0.008])
        arm.waitToDistination()

        # grab
        arm.enableSuctionCup(1)
        waitSecs(0.5)
        isGet = arm.isGrip()

        # move above
        arm.moveTo([c_xy_m[0], c_xy_m[1], 0.25])
        arm.waitToDistination()

        # reset
        arm.resetArmPosition()
        arm.waitToDistination()

        # release
        arm.enableSuctionCup(0)
        waitSecs(0.5)

        # remove out of box objs
        h_objs = removeOutOfBoxObjs(arm.clientID, h_objs, xBound2, yBound2)

arm.stopSim()




