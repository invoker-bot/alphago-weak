import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Dense, Flatten, Conv2D, Input, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adadelta
from godata import SgfDataBase, SgfFile, SgfSampleDataBase
from gotypes import GoIllegalActionError
import random
from typing import List
import numpy as np
from os import path, getcwd, makedirs
import time


class AlphaGo:
    def __init__(self, input_shape, dtype=np.float16, num_filters=128):
        """ probability"""

        _input = Input(shape=input_shape, name="network-input", dtype=dtype)

        x = Conv2D(num_filters, 5, padding="same", data_format="channels_first",
                   activation="relu")(_input)
        for _ in range(2, 7):
            x = Conv2D(num_filters, 3, padding="same", data_format="channels_first",
                       activation="relu")(x)
        policy_network_output = Conv2D(1, 1, padding='same',
                                       data_format='channels_first', activation='softmax')(x)
        policy_network_output = Flatten()(policy_network_output)
        policy_network_output = Reshape((input_shape[1], input_shape[2]), name="policy-network-output")(
            policy_network_output)

        value_network_output = Conv2D(num_filters, 3, padding='same',
                                      data_format='channels_first', activation='relu')(x)
        value_network_output = Conv2D(1, 1, padding='same',
                                      data_format='channels_first', activation='relu')(value_network_output)
        value_network_output = Flatten()(value_network_output)
        value_network_output = Dense(256, activation='relu')(value_network_output)

        value_network_output = Dense(1, activation='tanh', name="value-network-output")(value_network_output)

        self.model = Model(inputs=[_input], outputs=[policy_network_output, value_network_output])
        self.policy_network = Model(inputs=[_input], outputs=[policy_network_output], name="policy-network")
        self.policy_network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.value_network = Model(inputs=[_input], outputs=[value_network_output], name="value-network")
        self.value_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    @staticmethod
    def log_dir():
        return path.join(getcwd(), "log/models", time.strftime("%Y_%m_%d-%H_%M_%S"))


import tensorflow as tf

#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)

if __name__ == "__main__":
    with SgfSampleDataBase() as data:
        alpha = AlphaGo(input_shape=(9, 19, 19), dtype=np.float16)
        alpha.policy_network.fit(data.sample_generator(500),
                                 epochs=200, steps_per_epoch=500,
                                 validation_data=data.sample(10),
                                 callbacks=[
                                    EarlyStopping(patience=10),
                                    ModelCheckpoint("policy_network.h5", save_best_only=True),
                                    # TensorBoard(log_dir=AlphaGo.log_dir())
                                  ]
                                 )
    # alpha.policy_network.fit(X, Y, epochs=30, validation_split=0.2)
