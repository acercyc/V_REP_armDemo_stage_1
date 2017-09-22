import os
import numpy as np

try:
    from . import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')


# ============================================================================ #
#                               Generic Function                               #
# ============================================================================ #
def toLuaStr_array(num):
    sNum = [str(x) for x in num]
    LuaStr = ', '.join(sNum)
    return '{{{:s}}}'.format(LuaStr)


# ============================================================================ #
#                                 Function Call                                #
# ============================================================================ #
def loadApiDummy(clientID):
    path = os.path.dirname(__file__)
    return vrep.simxLoadModel(clientID, os.path.join(path, 'remoteApiDummy.ttm'), 1, vrep.simx_opmode_blocking)


def callAssociatedScriptFunction(clientID, objName, functionName, operationMode, *args):
    if len(args) > 0:
        argStr = ', ' + ', '.join(args)
    else:
        argStr = ''
    commandStr = "h = simGetScriptHandle('{:s}')\n" \
                 "returnVal = {{simCallScriptFunction('{:s}', h{:s})}}".format(objName, functionName, argStr)
    returns = vrep.simxCallScriptFunction(clientID,
                                          "remoteApiCommandServer", vrep.sim_scripttype_childscript,
                                          'executeCode_function_withReturnStr',
                                          [], [], [commandStr],
                                          bytearray(),
                                          operationMode)
    returnCode = returns[0]
    if len(returns[3]) > 0:
        returnStr = returns[3][0]
    else:
        returnStr = None
    return returnCode, returnStr, commandStr


def callBuildinFunction(clientID, functionName, operationMode, *args):
    argStr = ', '.join(args)
    commandStr = 'returnVal = {{{:s}({:s})}}'.format(functionName, argStr)
    returns = vrep.simxCallScriptFunction(clientID,
                                          "remoteApiCommandServer", vrep.sim_scripttype_childscript,
                                          'executeCode_function_withReturnStr',
                                          [], [], [commandStr],
                                          bytearray(),
                                          operationMode)
    returnCode = returns[0]
    if len(returns[3]) > 0:
        returnStr = returns[3][0]
    else:
        returnStr = None
    return returnCode, returnStr, commandStr


# ============================================================================ #
#                                  Controller                                  #
# ============================================================================ #
class VrepController:
    def __init__(self, ip='127.0.0.1', port=19997, timeOutInMs=5000, commThreadCycleInMs=5):
        self.ip = ip
        self.port = port
        self.timeOutInMs = timeOutInMs
        self.commThreadCycleInMs = commThreadCycleInMs
        self.clientID = self.connectToServer()
        self.h_ApiDummy = None
        self.loadApiDummy()

    # ---------------------------------------------------------------------------- #
    #                                                            Server Connection #
    # ---------------------------------------------------------------------------- #
    def connectToServer(self, ip=None, port=None, timeOutInMs=None, commThreadCycleInMs=None):
        if ip is None:
            ip = self.ip
        else:
            self.ip = ip

        if port is None:
            port = self.port
        else:
            self.port = port

        if timeOutInMs is None:
            timeOutInMs = self.timeOutInMs
        else:
            self.port = timeOutInMs

        if commThreadCycleInMs is None:
            commThreadCycleInMs = self.commThreadCycleInMs
        else:
            self.port = commThreadCycleInMs

        vrep.simxFinish(-1)  # just in case, close all opened connections
        clientID = vrep.simxStart(ip, port, True, True, timeOutInMs, commThreadCycleInMs)
        if clientID != -1:
            print('Connected to remote API server')
        else:
            print('Failed connecting to remote API server')
        return clientID

    def disconnectToServer(self):
        vrep.simxFinish(self.clientID)

    def resetConnection(self):
        vrep.simxFinish(-1)
        self.connectToServer()

    # ---------------------------------------------------------------------------- #
    #                                                                   Simulation #
    # ---------------------------------------------------------------------------- #
    def startSim(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

    def stopSim(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def restartSim(self):
        isRunning = True
        self.stopSim()
        while isRunning:
            isRunning = not (self.callBuildinFunction('simGetSimulationState', vrep.simx_opmode_blocking)[0])
        self.startSim()

    # ---------------------------------------------------------------------------- #
    #                                                                Function call #
    # ---------------------------------------------------------------------------- #
    def callAssociatedScriptFunction(self, objName, functionName, operationMode, *args):
        return callAssociatedScriptFunction(self.clientID, objName, functionName, operationMode, *args)

    def callBuildinFunction(self, functionName, operationMode, *args):
        return callBuildinFunction(self.clientID, functionName, operationMode, *args)

    # ---------------------------------------------------------------------------- #
    #                                                                    Api Dummy #
    # ---------------------------------------------------------------------------- #
    def loadApiDummy(self, isOverride=False):
        isNotExist, h_ApiDummy = vrep.simxGetObjectHandle(0, 'remoteApiCommandServer', vrep.simx_opmode_blocking)
        if isNotExist == 0:  # has existed
            if isOverride:
                vrep.simxRemoveObject(0, h_ApiDummy, vrep.simx_opmode_blocking)
            else:
                self.h_ApiDummy = h_ApiDummy
                return h_ApiDummy
        self.h_ApiDummy = loadApiDummy(self.clientID)
        return self.h_ApiDummy


# ============================================================================ #
#                                Data conversion                               #
# ============================================================================ #
def camDataToRGB(img, isZeroToOne=False):
    img = np.uint8(img.astype(np.int8))
    if isZeroToOne:
        img = img / 255.0
    return img