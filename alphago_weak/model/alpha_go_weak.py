# -*- coding: utf-8 -*-

import numpy as np
from os import path
from typing import *
import random
import os
import time
import math

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from functools import partial

from .basic import *
from ..board import *
from ..dataset import *
from ..board.board import GoBoard

__all__ = ["GoBoardAIV0", "AlphaGoWeakV0"]


class GoBoardAIV0(GoBoard):
    FEATURES = 17

    def __init__(self, size=19, dtype=np.float32):
        super().__init__(size)
        self.inputs = np.zeros((self.FEATURES, size, size), dtype=dtype)
        self.dtype = dtype
        self._next_player = GoPlayer.black

    def clean(self):
        super().clean()
        self.inputs = 0
        self._next_player = GoPlayer.black

    def update_inputs(self):
        self.inputs = np.roll(self.inputs, 2, axis=0)
        self.inputs[0] = 0 if self._next_player == GoPlayer.black else 1
        self.inputs[1] = self.grid == GoPlayer.black
        self.inputs[2] = self.grid == GoPlayer.white

    def play(self, point: Optional[GoPoint] = None, player=GoPlayer.none) -> Callable[[], None]:
        back = super().play(point, player)
        self._next_player = player.other
        self.update_inputs()
        return back

    def encode_policy_output(self, pos: Optional[GoPoint]) -> np.ndarray:
        """
        :return np.ndarray (362)
        """
        num = self._grid.shape[0] * self._grid.shape[1] + 1
        output = np.zeros(num, dtype=self.dtype)
        if pos is not None:
            output.itemset(pos.x * self._grid.shape[0] + pos.y, 1)
        else:
            output.itemset(-1, 1)
        return output

    def encode_value_output(self, winner: GoPlayer) -> np.ndarray:
        """
        :return np.ndarray (1) represents black or white
        """
        value = 1 if self._next_player == winner else 0 if winner == GoPlayer.none else -1
        return np.array(value, dtype=self.dtype)

    def valid_points(self, strings: List[GoString], dtype=np.float32) -> np.ndarray:
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


def game_dataset_generator(archive: GameArchive, size=19, dtype=np.float32):
    """
    """
    for data in archive:
        if data.size == size:
            b = GoBoardAIV0(data.size, dtype=dtype)
            b.setup_stones(*data.setup_stones)
            for player, pos in data.sequence:
                if pos is not None:
                    pos = GoPoint(*pos)
                yield b.inputs, (b.encode_policy_output(pos), b.encode_value_output(data.winner))
                b.play(pos, player)


TTreeNodeV0 = TypeVar("TTreeNodeV0", bound="TreeNodeV0")


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


class AlphaGoWeakV0(ModelBase, GoBoardAIV0):

    def __init__(self, name: str, root: str = None, size: int = 19, dtype=np.float32, num_filters=128):
        ModelBase.__init__(self, name, root)
        GoBoardAIV0.__init__(self, size)
        self.size = size
        self.dtype = dtype
        _input = Input(shape=(self.FEATURES, size, size), name="Input", dtype=dtype)
        conv = Conv2D(num_filters, 5, padding="same", data_format="channels_first",
                      activation="relu", name="Residual-Input")(_input)
        layer = conv
        for i in range(1, 12):
            layer = Conv2D(num_filters, 3, padding="same", data_format="channels_first",
                           activation="relu", name="Residual-Hidden" + str(i))(layer)
        residual = add([conv, layer], name="Residual-Output")
        layer = residual
        policy_network_output = Conv2D(1, 1, padding="same",
                                       data_format="channels_first", activation="softmax",
                                       name="Softmax")(residual)
        policy_network_output = Flatten(name="PolicyNetwork-Flatten")(
            policy_network_output)
        policy_network_output = Dense(size * size + 1, activation="softmax", name="PolicyNetwork-Output")(policy_network_output)
        value_network_output = Conv2D(num_filters, 3, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Input")(layer)
        value_network_output = Conv2D(1, 1, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Output")(value_network_output)
        value_network_output = Flatten(name="Flatten")(value_network_output)
        value_network_output = Dense(256, activation="relu", name="Relu")(value_network_output)
        value_network_output = Dense(1, activation="tanh", name="ValueNetwork-Output")(value_network_output)

        self.policy_network = Model(inputs=[_input], outputs=[policy_network_output], name="policy-network")
        self.policy_network.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        self.value_network = Model(inputs=[_input], outputs=[value_network_output], name="value-network")
        self.value_network.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        self.model = Model(inputs=[_input], outputs=[policy_network_output, value_network_output])
        self.model.compile(loss=["mean_squared_error", "mean_squared_error"], loss_weights=[0.4, 0.6],
                           optimizer="adam", metrics=["accuracy"])
        if path.exists(self.weights_path):
            self.load()

    def load(self):
        self.model.load_weights(self.weights_path)

    def save(self):
        self.model.save_weights(self.weights_path)

    def fit(self, archive: GameArchive, epochs=100, batch_size=512):
        dataset = tf.data.Dataset.from_generator(lambda: game_dataset_generator(archive, self.size, self.dtype),
                                                 output_signature=(tf.TensorSpec((self.FEATURES, self.size, self.size)),
                                                                   (tf.TensorSpec((self.size * self.size + 1,)), tf.TensorSpec(())))
                                                 ).batch(batch_size).prefetch(32).repeat(epochs)
        self.model.fit(dataset, callbacks=[
            TensorBoard(log_dir=self.logs_dir),
            EarlyStopping(patience=5),
            ModelCheckpoint(self.weights_path, save_best_only=True,
                            save_weights_only=True),
        ])

    def summary(self):
        self.model.summary()

    def plot(self, to_file: str = "model.png"):
        plot_model(self.model, to_file=to_file, show_shapes=True)

    def value_predict(self, player: GoPlayer) -> float:
        _input = self.encode_input(player, self.dtype)
        _input.shape = (1, *_input.shape)
        return self.value_network.predict(_input)[0, 0]

    def policy_predict(self, player: GoPlayer) -> np.ndarray:
        _input = self.encode_input(player, self.dtype)
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
