# -*- coding: utf-8 -*-

import os
import shutil
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
from ..utils.multi_works import do_cpu_intensive_works, do_works_experimental

__all__ = ["AlphaGoWeakV0", "AlphaGoWeak"]


class AlphaGoWeakV0(ModelBase):
    FEATURES = {
        "player_stone": 0,
        "opponent_stone": 1,
        "valid_point": 2,
        "player_stone_liberties": 3,
        "opponent_stone_liberties": 7,
        "length": 11
    }
    cached_steps = 4096

    @staticmethod
    def encode_input(player: GoPlayer, b_: GoBoardBase):
        input_ = np.zeros(
            (AlphaGoWeakV0.FEATURES["length"], b_.size, b_.size), dtype=np.float32)
        b_.encode(
            input_, AlphaGoWeakV0.FEATURES["player_stone"], GoBoardEncodeType.player_stone, player)
        b_.encode(input_, AlphaGoWeakV0.FEATURES["opponent_stone"],
                  GoBoardEncodeType.player_stone, player.other)
        b_.encode(
            input_, AlphaGoWeakV0.FEATURES["valid_point"], GoBoardEncodeType.valid_point, player)
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
        value_ = decay if player == winner else 0.0 if winner == GoPlayer.none else -decay
        return np.array(0.5 * (1 + value_), dtype=np.float32)

    def __init__(self, name: str = "alphago_weak_v0", root: str = None, size: int = 19, num_filters=128):
        ModelBase.__init__(self, name, root, size)
        self.cached: Dict[int, Tuple[np.ndarray, float]] = {}
        _input = Input(
            shape=(self.FEATURES["length"], size, size), name="Input", dtype=tf.float32)
        conv = Conv2D(num_filters, 5, padding="same", data_format="channels_first",
                      activation="relu", name="Residual-Input")(_input)
        layer = conv
        for i in range(1, int(size * 2 // 3)):
            layer = Conv2D(num_filters, 3, padding="same", data_format="channels_first",
                           activation="relu", name="Residual-Hidden" + str(i))(layer)
        residual = add([conv, layer], name="Residual-Output")
        layer = residual
        policy_network_output = Conv2D(1, 1, padding="same",
                                       data_format="channels_first", activation="softmax",
                                       name="Softmax")(residual)
        policy_network_output = Flatten(name="PolicyNetwork-Flatten")(
            policy_network_output)
        policy_network_output = Dense(
            size * size + 1, activation="softmax", name="PolicyOutput")(policy_network_output)
        value_network_output = Conv2D(num_filters, 3, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Input")(layer)
        value_network_output = Conv2D(1, 1, padding="same",
                                      data_format="channels_first", activation="relu",
                                      name="Relu-Output")(value_network_output)
        value_network_output = Flatten(name="Flatten")(value_network_output)
        value_network_output = Dense(
            256, activation="relu", name="Relu")(value_network_output)
        value_network_output = Dense(
            1, activation="sigmoid", name="ValueOutput")(value_network_output)
        self.model = Model(inputs=[_input], outputs=[
                           policy_network_output, value_network_output])
        self.model.compile(loss=["categorical_crossentropy", "mean_squared_error"], loss_weights=[0.7, 0.3],
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
    def _preprocess_archive_one(archive: GameArchive, counts: int, idx: int):
        I, P, V = [], [], []
        steps = AlphaGoWeakV0.cached_steps
        cache_dir = archive.cache_dir
        for i_ in range(idx, len(archive), counts):
            for i, (p, v) in AlphaGoWeakV0.game_data_generator(archive[i_]):
                I.append(i)
                P.append(p)
                V.append(v)
            if len(I) >= steps:
                np.save(path.join(cache_dir, f"{idx}.input"), np.array(
                    I[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.policy_output"), np.array(
                    P[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.value_output"), np.array(
                    V[:steps], dtype=np.float32))
                return True
        return False

    @staticmethod
    def preprocess_archive(archive: GameArchive, counts: int = None):
        if counts is None:
            counts = int(len(archive) * 128 / AlphaGoWeakV0.cached_steps)
        if path.exists(archive.cache_dir):
            shutil.rmtree(archive.cache_dir)
        makedirs(archive.cache_dir)
        finished_counts = sum(do_cpu_intensive_works(partial(AlphaGoWeakV0._preprocess_archive_one, archive, counts), list(
            range(counts)), total=counts, desc="Preprocess...", use_multiprocessing=True))
        print(f"successfully preprocessed: {finished_counts}/{counts}")

    @staticmethod
    def dataset_from_preprocess(archive: GameArchive):
        def np_load(idx, suffix):
            return np.load(path.join(archive.cache_dir, f"{int(idx)}.{suffix}.npy"))

        def py_generator(length: int):
            for i in range(length):
                I, P, V = np_load(i, "input"), np_load(
                    i, "policy_output"), np_load(i, "value_output")
                for row in range(AlphaGoWeakV0.cached_steps):
                    yield I[row], (P[row], V[row])

        length = len(os.listdir(archive.cache_dir)) // 3
        dataset = tf.data.Dataset.from_generator(lambda: py_generator(length),
                                                 output_signature=(tf.TensorSpec((AlphaGoWeakV0.FEATURES["length"], 19, 19)), (tf.TensorSpec((19 * 19 + 1,)), tf.TensorSpec(()))))
        return dataset, length * AlphaGoWeakV0.cached_steps

    def dataset_from_archive(self, archive: GameArchive):
        def py_func(idx):
            return tf.data.Dataset.from_generator(lambda idx: self.game_data_generator(archive[int(idx)]), args=(idx,),
                                                  output_signature=self.dataset_element_spec)

        dataset = tf.data.Dataset.range(len(archive)).interleave(
            py_func, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset, int(len(archive) * 128)

    def fit_from_dataset(self, dataset: tf.data.Dataset, steps: int, epochs=100, batch_size=512):
        self.model.fit(dataset.repeat(epochs).batch(batch_size).prefetch(512), steps_per_epoch=int(steps // batch_size), epochs=epochs,
                       callbacks=[
                           TensorBoard(log_dir=self.logs_dir),
                           EarlyStopping("loss", patience=5),
                           ModelCheckpoint(self.weights_path, "loss", save_best_only=True,
                                           save_weights_only=True),
        ])

    def fit_step_from_dataset(self, dataset: tf.data.Dataset) -> Tuple[float, float]:
        policy_accuracy = tf.keras.metrics.CategoricalAccuracy()
        loss_mean = tf.keras.metrics.Mean()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        loss_p = tf.keras.losses.CategoricalCrossentropy()
        loss_v = tf.keras.losses.MeanSquaredError()
        for x, (p, v) in dataset.batch(64):
            with tf.GradientTape() as tape:
                p_, v_ = self.model(x, training=True)
                loss = loss_p(p, p_) * 0.8 + loss_v(v, v_) * 0.2
                policy_accuracy.update_state(p, p_)
                loss_mean.update_state(loss)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
        return loss_mean.result(), policy_accuracy.result()

    def summary(self):
        self.model.summary()

    def plot(self, to_file: str = "model.png"):
        plot_model(self.model, to_file=to_file, show_shapes=True)

    def predict(self, board: GoBoardBase, player: GoPlayer):
        input_ = self.encode_input(player, board)
        input_.shape = (1, *input_.shape)
        policy_, value_ = self.model.predict(input_)
        policy_.shape = policy_.shape[1:]
        value_.shape = value_.shape[1:]
        return policy_, float(value_)

    def state_calculator(self, board: GoBoardBase, player: GoPlayer):
        policy_, value = self.predict(board, player)
        if policy_[-1] > 0.5:
            policy = [], []
        else:
            actions: List[GoPoint] = [pos for pos in board.valid_points(
                player) if board.eye_type(player, pos) < GoEyeType.unknown]
            weights: List[float] = [policy_.item(
                action.x * self.size + action.y) for action in actions]
            policy = actions, weights
        return policy, value

    @staticmethod
    def policy_evaluator(state, komi=6.5):
        return state[0]

    @staticmethod
    def value_evaluator(state: np.ndarray, komi=6.5):
        return state[1]


AlphaGoWeak = AlphaGoWeakV0
