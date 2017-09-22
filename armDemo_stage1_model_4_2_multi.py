# ============================================================================ #
#                    Create and train centre detection model                   #
# ============================================================================ #
# Acer 2017/09/22 14:19

from keras import optimizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model

from acerlib import keras_ext
from armDemo_stage1_loadData import loadData

# keras_ext.setTensorFlowGpuMemory(0.3)

# ============================================================================ #
#                                   Load Data                                  #
# ============================================================================ #
runID = 'run_multi_1'
dataPath = 'armDemo_vision_data_gen'

gaussFilter = 2
d_train = loadData(range(0, 18000), dataPath, runID, gaussFilter, True)

# ============================================================================ #
#                                  Build Model                                 #
# ============================================================================ #
input_shape = d_train[0][0].shape

l_input = Input(shape=input_shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(l_input)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(l_input)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(l_input)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (2, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='sigmoid')(x)
m = Model(l_input, x)
m.summary()

optm = optimizers.nadam(0.0001)
m.compile(optm, 'binary_crossentropy')

# keras_ext.plot_model(m)
# c = keras_ext.ModelController(m, 'armDemo_stage1', 'try_1')
# history = m.fit(npa(d_train[0]), npa(d_train[1]), batch_size=1, epochs=200, validation_data=d_val)

# ============================================================================ #
#                                   Training                                   #
# ============================================================================ #
p = keras_ext.Pipeline('armDemo_stage1', 'model_4_2')
p.m = m
p.d_train = d_train
# p.save()
p.fit(batch_size=128, epochs=1000, validation_split=0.2)
