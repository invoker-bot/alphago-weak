# -*- coding: utf-8 -*-

import os
import time
import math
import numpy as np
import tensorflow as tf
from functools import partial
from typing import *
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model

from .basic import *
from ..board import *
from ..dataset import *
from ..utils.multi_works import do_works

__all__ = ["AlphaGoWeakV0", "AlphaGoWeak"]

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


class AlphaGoWeakV0(ModelBase):
    FEATURES = {
        "player_stone": 0,
        "opponent_stone": 1,
        "valid_point": 2,
        "player_stone_liberties": 3,
        "opponent_stone_liberties": 7,
        "length": 11
    }
    cached_steps = 1024

    @staticmethod
    def encode_input(player: GoPlayer, b_: GoBoardBase):
        input_ = np.zeros((AlphaGoWeakV0.FEATURES["length"], b_.size, b_.size), dtype=np.float32)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["player_stone"], GoBoardEncodeType.player_stone, player)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["opponent_stone"], GoBoardEncodeType.player_stone, player.other)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["valid_point"], GoBoardEncodeType.valid_point, player)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["player_stone_liberties"], GoBoardEncodeType.player_stone_liberties, player,
                  4)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["opponent_stone_liberties"], GoBoardEncodeType.player_stone_liberties,
                  player.other, 4)
        return input_

    @staticmethod
    def encode_policy_output(size: int, pos: Optional[GoPoint]) -> np.ndarray:
        """
        Returns:
             array(362)
        """
        num = size * size + 1
        output = np.zeros(num, dtype=np.float32)
        if pos is not None:
            output.itemset(pos.x * size + pos.y, 1.0)
        else:
            output.itemset(-1, 1.0)
        return output

    @staticmethod
    def encode_value_output(player: GoPlayer, winner: GoPlayer, decay=1.0) -> np.ndarray:
        """
        Returns:
             array(1) represents black or white
        """
        value = 1.0 if player == winner else 0.0 if winner == GoPlayer.none else -1.0
        return np.array(decay * value, dtype=np.float32)

    def __init__(self, name: str = "alphago_weak_v0", root: str = None, size: int = 19, num_filters=128):
        ModelBase.__init__(self, name, root, size)
        _input = Input(shape=(self.FEATURES["length"], size, size), name="Input", dtype=tf.float32)
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
        policy_network_output = Dense(size * size + 1, activation="softmax", name="PolicyOutput")(policy_network_output)
        value_network_output = Conv2D(num_filters, 3, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Input")(layer)
        value_network_output = Conv2D(1, 1, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Output")(value_network_output)
        value_network_output = Flatten(name="Flatten")(value_network_output)
        value_network_output = Dense(256, activation="relu", name="Relu")(value_network_output)
        value_network_output = Dense(1, activation="tanh", name="ValueOutput")(value_network_output)

        self.policy_network = Model(inputs=[_input], outputs=[policy_network_output], name="policy-network")
        self.policy_network.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.value_network = Model(inputs=[_input], outputs=[value_network_output], name="value-network")
        self.value_network.compile(loss="mean_squared_error", optimizer="adam")
        self.model = Model(inputs=[_input], outputs=[policy_network_output, value_network_output])
        self.model.compile(loss=["categorical_crossentropy", "mean_squared_error"], loss_weights=[0.6, 0.4],
                           optimizer="adam", metrics=[['accuracy'], None])
        if path.exists(self.weights_path):
            self.load()

    def load(self):
        self.model.load_weights(self.weights_path)

    def save(self):
        self.model.save_weights(self.weights_path)

    @staticmethod
    def game_data_generator(data: GameData):
        b = GoBoard(data.size)
        b.setup_stones(*data.setup_stones)
        for steps, (player, pos) in enumerate(data.sequence):
            if pos is not None:
                pos = GoPoint(*pos)
            x = math.exp(- steps / (data.size * data.size))
            decay = min(2 * (1 - x) / (1 + x), 1.0)
            yield AlphaGoWeakV0.encode_input(player, b), (
                AlphaGoWeakV0.encode_policy_output(b.size, pos),
                AlphaGoWeakV0.encode_value_output(player, data.winner, decay))
            b.play(player, pos)

    @staticmethod
    def game_archive_generator(archive: GameArchive):
        for data in archive:
            yield from AlphaGoWeakV0.game_data_generator(data)

    @property
    def dataset_element_spec(self):
        return tf.TensorSpec((self.FEATURES["length"], self.size, self.size)), (tf.TensorSpec((self.size * self.size + 1,)), tf.TensorSpec(()))

    @staticmethod
    def _preprocess_archive_one(archive: GameArchive, cache_dir: str, idx: int, counts: int, steps=1024):
        I, P, V = [], [], []
        for i_ in range(idx, len(archive), counts):
            for i, (p, v) in AlphaGoWeakV0.game_data_generator(archive[i_]):
                I.append(i)
                P.append(p)
                V.append(v)
            if len(I) >= steps:
                np.save(path.join(cache_dir, f"{idx}.input"), np.array(I[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.policy_output"), np.array(P[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.value_output"), np.array(V[:steps], dtype=np.float32))
                return True
        return False

    def preprocess_archive(self, archive: GameArchive, counts: int = None):
        if counts is None:
            counts = int(len(archive) * 128 / self.cached_steps)
        do_works(partial(self._preprocess_archive_one, archive, self.cache_dir, counts=counts, steps=self.cached_steps), list(range(counts)), desc="Caching...")

    def dataset_from_preprocess(self):
        def np_load(idx, suffix):
            return np.load(path.join(self.cache_dir, f"{int(idx)}.{suffix}.npy"))

        def py_generator(length: int):
            for i in range(length):
                I, P, V = np_load(i, "input"), np_load(i, "policy_output"), np_load(i, "value_output")
                for row in range(self.cached_steps):
                    yield I[row], (P[row], V[row])

        length = len(os.listdir(self.cache_dir)) // 3
        dataset = tf.data.Dataset.from_generator(lambda: py_generator(length), output_signature=self.dataset_element_spec)
        return dataset, length * self.cached_steps

    def dataset_from_archive(self, archive: GameArchive):
        def py_func(idx):
            return tf.data.Dataset.from_generator(lambda idx: self.game_data_generator(archive[int(idx)]), args=(idx,),
                                                  output_signature=self.dataset_element_spec)

        dataset = tf.data.Dataset.range(len(archive)).interleave(py_func, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset, int(len(archive) * 128)

    def fit_from_dataset(self, dataset: tf.data.Dataset, steps: int, epochs=100, batch_size=512):
        self.model.fit(dataset.repeat(epochs).batch(batch_size).prefetch(512), steps_per_epoch=int(steps // batch_size), epochs=epochs,
                       callbacks=[
                           TensorBoard(log_dir=self.logs_dir),
                           EarlyStopping("loss", patience=5),
                           ModelCheckpoint(self.weights_path, save_best_only=True,
                                           save_weights_only=True),
                       ])

    def summary(self):
        self.model.summary()

    def plot(self, to_file: str = "model.png"):
        plot_model(self.model, to_file=to_file, show_shapes=True)

    def predict_value_once(self, player: GoPlayer, board: GoBoardBase) -> float:
        _input = self.encode_input(player, board)
        _input.shape = (1, *_input.shape)
        return self.value_network.predict(_input)[0, 0]

    def predict_policy_once(self, player: GoPlayer, board: GoBoardBase) -> np.ndarray:
        _input = self.encode_input(player, board)
        _input.shape = (1, *_input.shape)
        return self.policy_network.predict(_input)[0]

    def predict(self, board: GoBoardBase, player: GoPlayer, timeout: float = 1.0) -> Union[str, GoPoint]:
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


AlphaGoWeak = AlphaGoWeakV0
