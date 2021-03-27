# -*- coding: utf-8 -*-

import numpy as np
from os import path
from typing import *
import random
import os
import time
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from keras.layers import Dense, Flatten, Conv2D, Input, add, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from functools import partial

from ..basic import *
from .alpha_go import *

__all__ = ["GoBoardAIV0", "AlphaGoWeakV0"]


class GoBoardAIV0(GoBoard):
    Features = 7
    """
    Feature name            num of planes   Description
    Stone color             2               player stone / opponent 
    Valid position          1               Whether is a good position
    Liberties               4               Number of liberties (empty adjacent points) of the string of stones this move belongs to
    """
    Offset = {
        "stone_color": 0,
        "valid_position": 2,
        "liberties": 3,
    }

    def encode_input(self, player: GoPlayer, dtype=np.float16) -> np.ndarray:
        """
        :return  np.ndarray (7,size,size)
        """
        shape = self._grid.shape
        input_ = np.zeros((self.Features, shape[0], shape[1]), dtype=dtype)

        for pos in self:
            color = self._grid.item(pos)
            x, y = pos
            if color == player:
                input_.itemset((self.Offset["stone_color"], x, y), 1)
            elif color == player.other:
                input_.itemset((self.Offset["stone_color"] + 1, x, y), 1)

            if self.is_valid_point(pos) and not self.is_point_a_true_eye(pos):
                input_.itemset((self.Offset["valid_position"], x, y), 1)

        strings = self.get_strings()
        for string in strings:
            liberties = len(string.liberties)
            offset = 3 if liberties > 3 else liberties - 1
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
    def encode_value_output(player: GoPlayer, winner: GoPlayer, dtype=np.float16) -> np.ndarray:
        """
        :return np.ndarray (1) represents black or white
        """
        value = 1 if player == winner else 0.5 if winner == GoPlayer.none else 0
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


def sparse_sample_v0(data: GameData, dtype=np.float16):
    """
    """
    b = GoBoardAIV0(data.size)
    b.setup_stones(*data.setup_stones)
    steps = random.randint(0, len(data.sequence) - 1)
    for _step, (player, pos) in enumerate(data.sequence):
        if _step == steps:
            return b.encode_input(player, dtype=dtype), b.encode_policy_output(pos, dtype=dtype), b.encode_value_output(
                player, winner=data.winner, dtype=dtype)
        else:
            b.play(pos, player)
    return b.encode_input(GoPlayer.black, dtype=dtype), b.encode_policy_output(None,
                                                                               dtype=dtype), b.encode_value_output(
        GoPlayer.black, winner=data.winner, dtype=dtype)
    # raise IndexError("the length of GameData.sequence should not be 0")


class SampleDatabaseV0(ArrayDatabase):

    def __init__(self, name: str, size=19, dtype=np.float16):
        super().__init__("alphago_weak_v0", size)
        self.dtype = dtype
        self.name = name

    @staticmethod
    def _extract_one(key: str, size: int, dtype=np.float16):
        db = GameDatabase(size)
        data: GameData = db[key]
        return sparse_sample_v0(data, dtype=dtype)

    def extract(self, force=False):
        dtype = self.dtype
        database = GameDatabase(self.size)
        if force or self.name not in self:
            keys = database.keys()
            results = do_works(partial(SampleDatabaseV0._extract_one, size=self.size, dtype=dtype),
                               keys, desc="Extracting", unit="files")
            I, P, V = [], [], []
            for i, p, v in results:
                I.append(i)
                P.append(p)
                V.append(v)
            self[self.name] = np.array(I, dtype), np.array(P, dtype), np.array(V, dtype)

    def data(self):
        return self[self.name]


TTreeNodeV0 = TypeVar('TTreeNodeV0', bound='TreeNodeV0')


class TreeNodeV0:
    c = 1.5

    def __init__(self, parent: Optional[TTreeNodeV0], p):
        self.parent = parent
        self.children: Dict[GoPoint, TTreeNodeV0] = {}
        self.q = 0.0
        self.n = 0
        self.p = p

    def __setitem__(self, pos: GoPoint, node: TTreeNodeV0):
        self.children[pos] = node

    def __getitem__(self, pos: GoPoint):
        return self.children[pos]

    def __contains__(self, pos: GoPoint):
        return pos in self.children

    def __len__(self):
        return len(self.children)

    def __float__(self):
        return self.q + self.c * self.p * math.sqrt(self.n) / (self.n + 1)

    def __lt__(self, other):
        if isinstance(other, TreeNodeV0):
            return self.__float__() < other.__float__()
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, TreeNodeV0):
            return self.__float__() == other.__float__()
        return NotImplemented

    def max_child(self):
        return max(self.children.values())


class AlphaGoWeakV0(AlphaGoBase):
    name = "alphago_weak_v0"

    def __init__(self, size: int = 19, dtype=np.float16, num_filters=128):
        super().__init__(size=size, dtype=dtype)

        _input = Input(shape=(GoBoardAIV0.Features, size, size), name="Input", dtype=dtype)

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

    def init_fit(self, sample_name: str, weights_file: str,
                 epochs=100, batch_size=16,
                 validation_split=0.15, force=False):
        sample_database = SampleDatabaseV0(sample_name)
        sample_database.extract(force=force)
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

    def value_predict(self, board: GoBoardAIV0, player: GoPlayer) -> float:
        _input = board.encode_input(player, self.dtype)
        _input.shape = (1, *_input.shape)
        return self.value_network.predict(_input)[0, 0]

    def policy_predict(self, board: GoBoardAIV0, player: GoPlayer) -> np.ndarray:
        _input = board.encode_input(player, self.dtype)
        _input.shape = (1, *_input.shape)
        return self.policy_network.predict(_input)[0]

    def predict(self, board: GoBoardAIV0, player: GoPlayer, timeout: float = 1.0) -> Union[str, GoPoint]:
        start_time = time.perf_counter()
        # valid_positions = self.valid_positions(board, player, eps)
        weights = self.policy_predict(board, player)
        actions = [(0.0, "pass")]
        for i in range(self.size * self.size):
            pos = divmod(i, self.size)
            if board.is_valid_point(pos, player) and not board.is_point_a_true_eye(pos, player):
                if weights[i] > 0.003:
                    back = board.play(pos, player)
                    q = self.value_predict(board, player)
                    actions.append((q + self.size * weights[i], pos))
                    back()
        action = max(actions, key=lambda t: t[0])
        return action[1]
