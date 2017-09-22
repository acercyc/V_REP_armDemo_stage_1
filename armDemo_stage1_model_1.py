from acerlib.RemoteSession import findX11DisplayPort
findX11DisplayPort()

import os
import numpy as np
from numpy import array as npa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Lambda, Activation, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import LSTM, TimeDistributed, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras import backend
from keras import objectives

from acerlib import Pipeline, keras_ext
from armDemo_stage1_loadData import loadData


# ============================================================================ #
#                                   Load Data                                  #
# ============================================================================ #
runID = 'run_single_2'
dataPath = 'armDemo_vision_data_gen'

d_train, d_valid = loadData(range(0, 5000), dataPath, runID)
d_train = [npa(d_train[0]), npa(d_train[1])]
d_valid = [npa(d_valid[0]), npa(d_valid[1])]

# ============================================================================ #
#                                  Build Model                                 #
# ============================================================================ #
input_shape = d_train[0][0].shape
# input_shape = [110, 222, 3]

l_input = Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(l_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2DTranspose(512, (3, 4), activation='relu')(x)
x = Conv2DTranspose(512, (3, 4), activation='relu')(x)
x = Conv2DTranspose(512, (3, 4), activation='relu')(x)
x = UpSampling2D((2, 3))(x)
x = Conv2DTranspose(256, (3, 4), activation='relu')(x)
x = Conv2DTranspose(256, (3, 4), activation='relu')(x)
x = Conv2DTranspose(256, (3, 4), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(256, (3, 2), activation='relu')(x)
x = Conv2DTranspose(256, (3, 2), activation='relu')(x)
x = Conv2DTranspose(256, (3, 2), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(256, (3, 1), activation='relu')(x)
x = Conv2DTranspose(1, (1, 1), activation='relu')(x)
x = Reshape(input_shape[0:2])(x)

m = Model(l_input, x)
m.compile('Nadam', 'binary_crossentropy')

# keras_ext.plot_model(m)
# c = keras_ext.ModelController(m, 'armDemo_stage1', 'try_1')
# history = m.fit(npa(d_train[0]), npa(d_train[1]), batch_size=1, epochs=200, validation_data=d_val)

# ============================================================================ #
#                                   Training                                   #
# ============================================================================ #
p = keras_ext.Pipeline('armDemo_stage1', 'try_1')
p.m = m
# p.load_d_train()
# p.load_d_valid()
p.d_train = d_train
p.d_valid = d_valid
p.save()
p.fit(batch_size=32, epochs=500)
