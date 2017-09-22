# import matplotlib
# matplotlib.use('WX')

from acerlib.RemoteSession import findX11DisplayPort
findX11DisplayPort()

import importlib
from datetime import datetime
import os




import matplotlib.pyplot as plt
import numpy as np
from numpy import array as npa

from VrepPythonApi import vrep
from VrepPythonApi import vrep_ext
from armDemo import ArmDemo, waitSecs, ProxSensor_top
from acerlib import logging_ext

arm = ArmDemo('192.168.1.170')
arm.getImageFromCam(True, True)
plt.show()
# ================================ parameters ================================ #
# ============================================================================ #
#                               Connect to V-rep                               #
# ============================================================================ #
arm = ArmDemo('192.168.1.170')

# restart simulation
arm.restartSim()
arm.createRandomObj(30)

arm.takePicture('temp.png')/255


h_cam = vrep.simxGetObjectHandle(0, 'boxCam', vrep.simx_opmode_blocking)
dimg = vrep.simxGetVisionSensorImage(0, h_cam[1], 0, vrep.simx_opmode_blocking)

ii = npa(dimg[2]).reshape([256, 256, 3])


def camDataToRGB(img, isZeroToOne=False):
    img = np.uint8(img.astype(np.int8))
    if isZeroToOne:
        img = img / 255.0
    return img


iii = camDataToRGB(ii, True)
plt.figure()
plt.ion()
plt.imshow(iii)
plt.show()




# ============================================================================
plt.figure()
d = arm.getImageFromCam()/1.0
plt.ion()
plt.imshow(d, vmin=-127, vmax=128)
plt.show()

# ----------------------------------------------------------------------------
d = -arm.getImageFromCam()/2
plt.ion()
plt.imshow(d, vmin=-127, vmax=128)
plt.show()

# ----------------------------------------------------------------------------
d = (arm.getImageFromCam()-127.0)/256.0
d[0:10, 0, 0]

# ============================================================================
plt.ion()
img = plt.imread('temp.png')
plt.imshow(img)
plt.show()


img[0:10, 0, 0]

# ------------------------------------------------------------------
from scipy.interpolate import interp1d

f = interp1d([0, 255], [0, 1])
f(208)

np.uint8(np.int8(-48))

# ----------------------------------------------------------------------------
d = arm.getImageFromCam(False, True)