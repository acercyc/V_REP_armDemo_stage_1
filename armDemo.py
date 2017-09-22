# ============================================================================ #
#                           Modified for stage1 demo                           #
# ============================================================================ #
#
# Add proxSensor 2
#   1.1 - Acer 2017/08/31 12:38


import numpy as np
import matplotlib.pyplot as plt
from numpy import array as npa
from datetime import datetime

from VrepPythonApi import vrep
from VrepPythonApi import vrep_ext


def waitSecs(t):
    t0 = datetime.now()
    dt = (datetime.now() - t0).total_seconds()
    while dt < t:
        dt = (datetime.now() - t0).total_seconds()


class ArmDemo(vrep_ext.VrepController):
    def __init__(self, *args, **kwargs):
        super(ArmDemo, self).__init__(*args, **kwargs)
        self.h_orig = vrep.simxGetObjectHandle(self.clientID, 'box_orig', vrep.simx_opmode_blocking)[1]
        self.h_arm = vrep.simxGetObjectHandle(self.clientID, 'redundantRob_target', vrep.simx_opmode_blocking)[1]
        self.h_movingLagDist = vrep.simxGetDistanceHandle(self.clientID, 'movingLagDist', vrep.simx_opmode_blocking)[1]
        self.h_proxSensor2 = \
            vrep.simxGetObjectHandle(self.clientID, 'uarmVacuumGripper_sensor2', vrep.simx_opmode_blocking)[1]
        self.position_init = self.getArmPosition()

    def startSim(self):
        super(ArmDemo, self).startSim()
        self.position_init = self.getArmPosition()

    def moveTo(self, targetPosition):
        vrep.simxSetObjectPosition(self.clientID, self.h_arm, self.h_orig, targetPosition,
                                   vrep.simx_opmode_blocking)

        # --- old method: using v-rep scene model --- #
        # targetPosition = vrep_ext.toLuaStr_array(targetPosition)
        # self.callAssociatedScriptFunction('redundantRobot', 'moveTo',
        #                                   vrep.simx_opmode_blocking, targetPosition)
        # ---------------------------------------------------------------------------- #

    def enableSuctionCup(self, isOn):
        if isOn:
            self.callAssociatedScriptFunction('redundantRobot', 'enableSuctionCup', vrep.simx_opmode_blocking, 'true')
        else:
            self.callAssociatedScriptFunction('redundantRobot', 'enableSuctionCup', vrep.simx_opmode_blocking, 'false')

    def isGrip(self):
        h_grabDummyParent = vrep.simxGetObjectHandle(self.clientID,
                                                     'uarmVacuumGripper_link2_dyn',
                                                     vrep.simx_opmode_blocking)[1]
        h_grabDummy = vrep.simxGetObjectHandle(self.clientID,
                                               'uarmVacuumGripper_loopDummyA',
                                               vrep.simx_opmode_blocking)[1]
        h_grabDummyParentNow = vrep.simxGetObjectParent(self.clientID, h_grabDummy, vrep.simx_opmode_blocking)[1]
        return not (h_grabDummyParent == h_grabDummyParentNow)

    def objCreation(self, position, ori, isVisible=True):
        position = vrep_ext.toLuaStr_array(position)
        ori = vrep_ext.toLuaStr_array(ori)
        h_obj = self.callAssociatedScriptFunction('box', 'objCreation', vrep.simx_opmode_blocking, position, ori)
        h_obj = int(h_obj[1])
        if not isVisible:
            vrep.simxSetObjectIntParameter(self.clientID, h_obj, 10, 0, vrep.simx_opmode_blocking)
        return h_obj

    def setObjVisibility(self, h_objs, isVisible):
        for h in h_objs:
            if isVisible:
                vrep.simxSetObjectIntParameter(self.clientID, h, 10, 1, vrep.simx_opmode_blocking)
            else:
                vrep.simxSetObjectIntParameter(self.clientID, h, 10, 0, vrep.simx_opmode_blocking)

    def createRandomObj(self, n, isVisible=True):
        X = np.random.uniform(0, 1, n)
        Y = np.random.uniform(0, 0.5, n)
        Z = np.random.uniform(0, 0.2, n)
        ori_Z = np.random.uniform(0, 90, n)
        h_objs = []
        for x, y, z, ori_z in zip(X, Y, Z, ori_Z):
            h_objs.append(self.objCreation([x, y, z], [0, 0, ori_z], isVisible))
        return h_objs

    def removeObj(self, h_objs):
        for i in h_objs:
            vrep.simxRemoveObject(self.clientID, i, vrep.simx_opmode_blocking)

    def getObjPosition(self, h_objs):
        positions = []
        for i in h_objs:
            positions.append(vrep.simxGetObjectPosition(self.clientID, i, self.h_orig, vrep.simx_opmode_blocking)[1])

        return positions

    def getImageFromCam(self, showImg=False, isZeroToOne=False):
        h_cam = vrep.simxGetObjectHandle(self.clientID, 'boxCam', vrep.simx_opmode_blocking)
        dimg = vrep.simxGetVisionSensorImage(self.clientID, h_cam[1], 0, vrep.simx_opmode_blocking)
        img = np.reshape(npa(dimg[2]), [dimg[1][0], dimg[1][0], 3])
        img = np.flipud(img)
        img = vrep_ext.camDataToRGB(img, isZeroToOne=isZeroToOne)
        if showImg:
            plt.imshow(img)
        return img

    def takePicture(self, filename, showImg=False):
        img = self.getImageFromCam(showImg)
        plt.imsave(filename, img)
        if showImg:
            plt.imshow(img)
        return img

    def getArmPosition(self):
        position = vrep.simxGetObjectPosition(self.clientID, self.h_arm, self.h_orig, vrep.simx_opmode_blocking)
        return position[1]

    def resetArmPosition(self):
        vrep.simxSetObjectPosition(self.clientID, self.h_arm, self.h_orig, self.position_init,
                                   vrep.simx_opmode_blocking)

    def getMovingLagDist(self):
        movingLagDist = vrep.simxReadDistance(self.clientID, self.h_movingLagDist, vrep.simx_opmode_blocking)[1]
        return movingLagDist

    def waitToDistination(self, minDist=0.004, timeOut=1):
        isReach = 0
        t0 = datetime.now()
        dt = (datetime.now() - t0).total_seconds()

        while (not isReach) and (dt < timeOut):
            isReach = self.getMovingLagDist() <= minDist
            dt = (datetime.now() - t0).total_seconds()
        return isReach

    def getProxSensorData(self):
        dist = vrep.simxReadProximitySensor(self.clientID, self.h_proxSensor2, vrep.simx_opmode_blocking)[2][2]
        return dist


class ProxSensor_top:
    def __init__(self, clientID):
        self.clientID = clientID
        self.h = vrep.simxGetObjectHandle(clientID, 'pSensor_top', vrep.simx_opmode_blocking)[1]
        self.h_orig = vrep.simxGetObjectHandle(clientID, 'box_orig', vrep.simx_opmode_blocking)[1]

    def getData(self):
        p_obj = vrep.simxReadProximitySensor(self.clientID, self.h, vrep.simx_opmode_blocking)[2]
        p_self = vrep.simxGetObjectPosition(self.clientID, self.h, self.h_orig, vrep.simx_opmode_blocking)[1]
        p = [p_self[0] - p_obj[0], p_self[1] + p_obj[1], p_self[2] - p_obj[2]]
        return p
