# -*- coding: utf-8 -*-

import numpy as np
from os import path, cpu_count
from typing import *
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from keras.layers import Dense, Flatten, Conv2D, Input, add, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import tqdm

from ..basic import *
from .alpha_go import *

__all__ = ["GoBoardAI", "AlphaGoWeak"]


class GoBoardAI(GoBoard):
    Features = 9
    """
    Feature name            num of planes   Description
    Stone color            3               black / white / next player
    Valid position          1               Whether is a valid position
    Sensibleness            1               Whether a move does not fill its own eyes
    Liberties               4               Number of liberties (empty adjacent points) of the string of stones this move belongs to
    """
    Offset = {
        "stone_color_black": 0,
        "stone_color_white": 1,
        "next_player_color": 2,
        "valid_position": 3,
        "sensibleness": 4,
        "liberties": 5,
    }

    def encode_input(self, dtype=np.float16) -> np.ndarray:
        """
        :return  np.ndarray (9,size,size)
        """
        shape = self._grid.shape
        input_ = np.zeros((self.Features, shape[0], shape[1]), dtype=dtype)

        for pos in self.__iter__():
            player = self.__getitem__(pos)
            x, y = pos
            if player == GoPlayer.black:
                input_.itemset((self.Offset["stone_color_black"], x, y), 1)
            elif player == GoPlayer.white:
                input_.itemset((self.Offset["stone_color_white"], x, y), 1)

            if player == self._next_player:
                input_.itemset((self.Offset["next_player_color"], x, y), 1)

            if self.is_point_a_true_eye(pos):
                input_.itemset((self.Offset["sensibleness"], x, y), 1)
        strings = self.get_strings()
        input_[self.Offset["valid_position"]] = self.valid_points(strings, dtype)
        for string in strings:
            liberties = len(string.liberties)
            offset = 3 if liberties >= 4 else liberties - 1
            for stone in string.stones:
                input_.itemset((self.Offset["liberties"] + offset, *stone), 1)
        return input_

    def encode_policy_output(self, pos: Optional[GoPoint], dtype=np.float16) -> np.ndarray:
        """
        :return np.ndarray (361)
        """
        num = self._grid.shape[0] * self._grid.shape[1]
        output = np.zeros(num, dtype=dtype)
        if pos is not None:
            output.itemset(pos[0] * self._grid.shape[0] + pos[1], 1)
        else:
            output += 1.0 / (self._grid.shape[0] * self._grid.shape[1])
        return output

    @staticmethod
    def encode_value_output(player: GoPlayer, dtype=np.float16) -> np.ndarray:
        """
        :return np.ndarray (1) represents black or white
        """
        value = 1 if player == GoPlayer.black else 0.5 if player == GoPlayer.none else 0
        return np.array(value, dtype=dtype)

    def valid_points(self, strings: List[GoString], dtype=np.float16) -> np.ndarray:
        tensor = np.zeros(self._grid.shape, dtype=dtype)
        for pos in self.__iter__():
            if self._grid.item(pos) == GoPlayer.none.value:
                tensor.itemset(pos, 1)
        confuses = set()
        for string in strings:
            if len(string.liberties) == 1:
                confuses.update(string.liberties)
        for confuse in confuses:
            if self.is_valid_point(confuse):
                tensor.itemset(confuse, 1)
            else:
                tensor.itemset(confuse, 0)
        return tensor


def sparse_sample(data: GameData, dtype=np.float16):
    """
    """
    b = GoBoardAI(data.size)
    b.setup_stones(*data.setup_stones)
    steps = random.randint(0, len(data.sequence) - 1)
    for _step, (player, pos) in enumerate(data.sequence):
        if _step == steps:
            return b.encode_input(dtype=dtype), b.encode_policy_output(pos, dtype=dtype), b.encode_value_output(
                data.winner, dtype=dtype)
        else:
            b.play(pos, player)
    return b.encode_input(dtype=dtype), b.encode_policy_output((0, 0), dtype=dtype), b.encode_value_output(
        data.winner, dtype=dtype)
    # raise IndexError("the length of GameData.sequence should not be 0")


class SampleDatabase(ArrayDatabase):

    def __init__(self, name: str, size=19, dtype=np.float16):
        super().__init__("alphago_weak", size)
        self.dtype = dtype
        self.name = name

    @staticmethod
    def _extract_one(key: str, size: int, dtype=np.float16):
        db = GameDatabase(size)
        data: GameData = db[key]
        return sparse_sample(data, dtype=dtype)

    def extract(self, force=False, multiprocess=False):
        dtype = self.dtype
        database = GameDatabase(self.size)
        if force or self.name not in self:
            keys = database.keys()
            bar = tqdm.tqdm(desc="Extract...", total=len(keys), unit="files", unit_scale=True)
            with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
                I, P, V = [], [], []
                while len(keys) > 0:
                    processing_keys = keys[:10000]
                    keys = keys[10000:]
                    for i, p, v in executor.map(partial(SampleDatabase._extract_one, size=self.size, dtype=dtype),
                                                processing_keys):
                        I.append(i)
                        P.append(p)
                        V.append(v)
                        bar.update(1)
                self[self.name] = np.array(I, dtype), np.array(P, dtype), np.array(V, dtype)

    def data(self):
        return self[self.name]


class AlphaGoWeak(AlphaGoBase):

    def __init__(self, size: int = 19, dtype=np.float16, num_filters=128):
        super().__init__("alphago_weak", size=size, dtype=dtype)

        _input = Input(shape=(GoBoardAI.Features, size, size), name="Input", dtype=dtype)

        conv = Conv2D(num_filters, 5, padding="same", data_format="channels_first",
                      activation="relu", name="Residual-Input")(_input)
        layer = conv
        for i in range(1, 6):
            layer = Conv2D(num_filters, 3, padding="same", data_format="channels_first",
                           activation="relu", name="Residual-Hidden" + str(i))(layer)

        residual = add([conv, layer], name="Residual-Output")
        layer = residual

        policy_network_output = Conv2D(1, 1, padding='same',
                                       data_format='channels_first', activation='softmax',
                                       name="Softmax")(
            residual)
        policy_network_output = Flatten(name="PolicyNetwork-Flatten")(
            policy_network_output)

        policy_network_output = Activation("softmax", name="PolicyNetwork-Output")(policy_network_output)

        value_network_output = Conv2D(num_filters, 3, padding='same',
                                      data_format='channels_first', activation='relu',
                                      name="Relu-Input")(layer)
        value_network_output = Conv2D(1, 1, padding='same',
                                      data_format='channels_first', activation='relu',
                                      name="Relu-Output")(value_network_output)
        value_network_output = Flatten(name="Flatten")(value_network_output)
        value_network_output = Dense(256, activation='relu', name="Relu")(value_network_output)

        value_network_output = Dense(1, activation='tanh', name="ValueNetwork-Output")(value_network_output)

        self.policy_network = Model(inputs=[_input], outputs=[policy_network_output], name="policy-network")
        self.policy_network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.value_network = Model(inputs=[_input], outputs=[value_network_output], name="value-network")
        self.value_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        self.model = Model(inputs=[_input], outputs=[policy_network_output, value_network_output])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], loss_weights=[0.4, 0.6],
                           optimizer='sgd', metrics=['accuracy'])

    def load(self, file: str = "default.h5"):
        self.model.load_weights(self.weights_file(file))

    def save(self, file: str = "default.h5"):
        self.model.save_weights(self.weights_file(file))

    def fit(self, sample_database: SampleDatabase, weights_file: str,
            epochs=100, batch_size=16,
            validation_split=0.15, force=False):
        if force or not path.exists(self.weights_file(weights_file)):
            X, P, V = sample_database.data()
            self.model.fit(x=X, y=(P, V), batch_size=batch_size, epochs=epochs, callbacks=[
                TensorBoard(log_dir=self.log_dir()),
                EarlyStopping(patience=5),
                ModelCheckpoint(self.weights_file(weights_file), save_best_only=True,
                                save_weights_only=True),
            ],
                           validation_split=validation_split)

    def summary(self):
        self.model.summary()

    def plot(self, file: str):
        plot_model(self.model, to_file=file, show_shapes=True)

    def value_predict(self, board: GoBoardAI) -> float:
        _input = board.encode_input(self.dtype)
        _input.shape = (1, *_input.shape)
        return self.value_network.predict(_input)[0, 0]

    def policy_predict(self, board: GoBoardAI) -> np.ndarray:
        _input = board.encode_input(self.dtype)
        _input.shape = (1, *_input.shape)
        return self.policy_network.predict(_input)[0]


if __name__ == "__main__":
    set_cache_dir()
    s = SampleDatabase("sparse_data")
    s.extract()
    al = AlphaGoWeak()
    al.plot("img/model.png")
    al.fit(s, "default.h5")
