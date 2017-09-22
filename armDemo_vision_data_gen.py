# ============================================================================ #
#                     Generate pictures with random objects                    #
# ============================================================================ #
import os
import numpy as np
from numpy import array as npa
from armDemo import ArmDemo, waitSecs

# ================================ parameters ================================ #
nObj = 10
nSample = 50000
pathname = 'armDemo_vision_data_gen'
pathname_run = 'run_multi_1'

# ============================== initialisation ============================== #
try:
    os.mkdir(pathname)
except:
    print('Folder exists')

try:
    os.mkdir(os.path.join(pathname, pathname_run))
except:
    print('Folder exists')

# ============================================================================ #
#                               Connect to V-rep                               #
# ============================================================================ #
arm = ArmDemo('192.168.1.170')

# restart simulation
arm.restartSim()
data = []

for i in range(nSample):
    print(i)

    # create objects
    h_objs = arm.createRandomObj(nObj)
    waitSecs(2)

    # get position
    pos = arm.getObjPosition(h_objs)
    pos = npa(pos)
    pos_data = np.hstack([pos, np.ones([pos.shape[0], 1]) * i])
    data.append(pos_data)

    # take picture
    fName = os.path.join(pathname, pathname_run, '{:d}.png'.format(i))
    arm.takePicture(fName)
    waitSecs(0.5)

    # remove object
    arm.removeObj(h_objs)
    waitSecs(1)

    # save data
    data_array = np.vstack(data)
    fName = os.path.join(pathname, '{:s}.txt'.format(pathname_run))
    np.savetxt(fName, data_array)
