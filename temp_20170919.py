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
from acerlib import logging_ext, keras_ext

from keras.models import load_model


arm = ArmDemo('192.168.1.170', commThreadCycleInMs=1)

arm.restartSim()

for i in range(20):
    arm.moveTo([0.8, 0.3, 0.25])
    arm.waitToDistination()

    arm.resetArmPosition()
    arm.waitToDistination()