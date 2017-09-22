# ============================================================================ #
#                     A class for keras pipeline processing                    #
# ============================================================================ #
# Need to specify path when initialising a new pipeline object

# 1.0 - Acer 2017/02/14 18:58
# 2.0 - Acer 2017/02/15 19:04
# 3.0 - Acer 2017/02/17 15:58
# 3.1 - Acer 2017/03/01 16:22
# 3.2 - Acer 2017/04/26 16:02
# 3.3 - Acer 2017/04/28 16:56
# 3.4 - Acer 2017/05/01 20:41
# 3.5 - Acer 2017/05/15 12:16


import os
from subprocess import Popen
import time
import inspect
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils.vis_utils import plot_model

import acerlib.shelve_ext as she
from acerlib import sequence


class Pipeline:
    # ============================================================================ #
    #                                  Initialise                                  #
    # ============================================================================ #
    def __init__(self, path='pipeline_temp', ID=None):
        if ID is None:
            ID = 'p' + time.strftime("%Y%d%d_%H%M%S")
        self.ID = ID
        # create data folder
        self.path = path
        self.check_and_create_path()

        # data
        self.d_train = None  # should be a list [X, Y]
        self.d_test = None  # should be a list [X, Y]
        self.d_valid = None  # should be a list [X, Y]

        # model
        self.m = None

        # fitting
        self.history = None

    # ============================================================================ #
    #                             High Level Functions                             #
    # ============================================================================ #
    # Training ------------------------------------------------------------------- #
    def fit(self, batch_size=32, epochs=10, callbacks=None, validation_split=0.0, useValidation_data=True):
        if callbacks is None:
            callbacks = self.defaultCallbacks()
        if useValidation_data:
            d_valid = self.d_valid
        else:
            d_valid = None
        self.history = self.m.fit(self.d_train[0], self.d_train[1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_split=validation_split,
                                  validation_data=d_valid)

    # ============================================================================ #
    #                                     Plot                                     #
    # ============================================================================ #
    def plot_m(self):
        fName = os.path.join(self.path, '%s_plot_m.png' % self.ID)
        plot_model(self.m, show_shapes=True, to_file=fName)
        img = mpimg.imread(fName)
        try:
            Popen(['eog', fName])
        except:
            plt.imshow(img)

    def plot_history(self):
        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.ID)
        history = np.loadtxt(fName, skiprows=1, delimiter=',')
        plt.ion()
        plt.figure(figsize=(13, 5))
        plt.plot(history[:, 1], label="training")
        plt.plot(history[:, 2], label="validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    # ============================================================================ #
    #                                   callback                                   #
    # ============================================================================ #
    def defaultCallbacks(self):

        # save the best model
        fName = os.path.join(self.path, '%s_CModelCheckpoint_best.hdf5' % self.ID)
        cb_ModelCheckpoint_best = ModelCheckpoint(fName, monitor='val_loss', save_best_only=True)

        # log history
        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.ID)
        cb_CSVLogger = CSVLogger(fName)

        # save all model history
        pathName = os.path.join(self.path, 'model_history_%s' % self.ID)
        self.check_and_create_path(pathName)

        fName = os.path.join(pathName, '%s_CModelCheckpoint_{epoch:04d}.hdf5' % self.ID)
        cb_ModelCheckpoint = ModelCheckpoint(fName, monitor='val_loss')

        return [cb_ModelCheckpoint_best, cb_CSVLogger, cb_ModelCheckpoint]

    # ============================================================================ #
    #                                   File I/O                                   #
    # ============================================================================ #
    def load(self, m=True, d_train=True, d_test=True, d_valid=True):
        funMappting = {'m': [m, self.load_m],
                       'd_train': [d_train, self.load_d_train],
                       'd_test': [d_test, self.load_d_test],
                       'd_valid': [d_valid, self.load_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:
                    fun[1]()
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not loaded\n')
        print('')

    def save(self, m=True, d_train=True, d_test=True, d_valid=True):
        funMappting = {'m': [m, self.save_m],
                       'd_train': [d_train, self.save_d_train],
                       'd_test': [d_test, self.save_d_test],
                       'd_valid': [d_valid, self.save_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:  # if data exist, then save
                    if getattr(self, key) is not None:
                        fun[1]()  # run save funciton
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not saved\n')
        print('')
        self.save_pipeline()

    def read_d(self, d_train=True, d_test=True, d_valid=True):
        funMappting = {'d_train': [d_train, self.read_d_train],
                       'd_test': [d_test, self.read_d_test],
                       'd_valid': [d_valid, self.read_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:
                    fun[1]()
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not read\n')
        print('')

    # ============================================================================ #
    #                              Low-level File I/O                              #
    # ============================================================================ #

    # save ----------------------------------------------------------------------- #
    def save_d_train(self):
        fName = os.path.join(self.path, '%s_d_train.npz' % self.ID)
        np.savez(fName, *self.d_train)
        print('training data saved')

    def save_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.ID)
        np.savez(fName, *self.d_valid)
        print('validation data saved')

    def save_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.ID)
        np.savez(fName, *self.d_test)
        print('testing data saved')

    def save_m(self):
        fName = os.path.join(self.path, '%s_m' % self.ID)
        self.m.save(fName)
        print('model saved')

    def save_pipeline(self):
        ps = copy.copy(self)
        ps.d_train = []
        ps.d_test = []
        ps.d_valid = []
        ps.m = []

        fName = os.path.join(self.path, '%s_pipeline' % self.ID)
        she.save(fName, 'pipeline', ps)
        print('Pipeline saved')

    # load ----------------------------------------------------------------------- #
    def load_d_train(self):
        fName = os.path.join(self.path, '%s_d_train.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_train = [d[vName] for vName in d.files]
        print('trainig data loaded')

    def load_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_test = [d[vName] for vName in d.files]
        print('testing data loaded')

    def load_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_valid = [d[vName] for vName in d.files]
        print('validation data loaded')

    def load_m(self):
        fName = os.path.join(self.path, '%s_m' % self.ID)
        load_model(fName)
        print('model loaded')

    # ============================================================================ #
    #                                   Utilities                                  #
    # ============================================================================ #
    def check_and_create_path(self, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)


def load_pipeline(fName):
    p = she.load(fName, 'pipeline')
    p.load()
    return p


def load_pipeline_withBestModel(fName):
    p = she.load(fName, 'pipeline')
    p.load()
    fName = os.path.join(p.path, '%s_CModelCheckpoint_best.hdf5' % p.id)
    p.m = load_model(fName)
    return p

class modelController:
    # pipeline info
    path = None
    id = None
    loss = None
    loss_best = np.inf

    def __init__(self, m, path='pipeline_temp', ID=None):
        # create data folder
        self.m = m
        self.path = path
        self.check_and_create_path()
        if ID is None:
            self.id = 'p' + time.strftime("%Y%d%d_%H%M%S")
        else:
            self.id = ID

    def check_and_create_path(self, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)

    def save_m(self):
        fName = os.path.join(self.path, '%s_m' % self.id)
        self.m.save(fName)
        print('model saved')

    def save_m_best(self, newLoss):
        if newLoss < self.loss_best:
            self.loss_best = newLoss
            fName = os.path.join(self.path, '%s_m_best' % self.id)
            self.m.save(fName)
            print('model saved')

    # def wrtie_loss(self, newLoss):
    #     fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.id)
    #     with open(fName, "a") as myfile:
    #         myfile.write("appended text", )

    def plot_m(self):
        fName = os.path.join(self.path, '%s_plot_m.png' % self.id)
        plot_model(self.m, show_shapes=True, to_file=fName)
        img = mpimg.imread(fName)
        try:
            Popen(['eog', fName])
        except:
            plt.imshow(img)
