# -*- coding: utf-8 -*-

import math
import os
import shutil
from functools import partial
from os import makedirs, path
from typing import *

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .basic import ModelBase
from ..board import *
from ..dataset import *
from ..utils.multi_works import do_cpu_intensive_works

__all__ = ["AlphaGoWeakV0", "AlphaGoWeak"]


class _AlphaGoWeakNet(nn.Module):
    def __init__(self, size: int, input_planes: int, num_filters: int = 128):
        super().__init__()
        self.size = size
        self.residual_input = nn.Conv2d(input_planes, num_filters, 5, padding=2)
        hidden_layers = max(int(size * 2 // 3) - 1, 0)
        self.residual_hidden = nn.ModuleList(
            nn.Conv2d(num_filters, num_filters, 3, padding=1) for _ in range(hidden_layers)
        )
        self.policy_conv = nn.Conv2d(num_filters, 1, 1)
        self.policy_output = nn.Linear(size * size, size * size + 1)
        self.value_conv_input = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.value_conv_output = nn.Conv2d(num_filters, 1, 1)
        self.value_hidden = nn.Linear(size * size, 256)
        self.value_output = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_input = F.relu(self.residual_input(x))
        layer = residual_input
        for conv in self.residual_hidden:
            layer = F.relu(conv(layer))
        residual = residual_input + layer

        policy = self.policy_conv(residual)
        policy = torch.flatten(policy, start_dim=1)
        policy = F.softmax(self.policy_output(policy), dim=1)

        value = F.relu(self.value_conv_input(residual))
        value = F.relu(self.value_conv_output(value))
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.value_hidden(value))
        value = torch.sigmoid(self.value_output(value)).squeeze(-1)
        return policy, value


class _PreprocessedGameDataset(IterableDataset):
    def __init__(self, cache_dir: str, cached_steps: int):
        self.cache_dir = cache_dir
        self.cached_steps = cached_steps

    def __iter__(self):
        length = len(os.listdir(self.cache_dir)) // 3
        for idx in range(length):
            inputs = np.load(path.join(self.cache_dir, f"{idx}.input.npy"))
            policies = np.load(path.join(self.cache_dir, f"{idx}.policy_output.npy"))
            values = np.load(path.join(self.cache_dir, f"{idx}.value_output.npy"))
            for row in range(self.cached_steps):
                yield inputs[row], (policies[row], values[row])


class _ArchiveGameDataset(IterableDataset):
    def __init__(self, archive: GameArchive):
        self.archive = archive

    def __iter__(self):
        for data in self.archive:
            yield from AlphaGoWeakV0.game_data_generator(data)


class _ArrayGameDataset(Dataset):
    def __init__(self, inputs: Sequence[np.ndarray], policies: Sequence[np.ndarray], values: Sequence[np.ndarray]):
        if not (len(inputs) == len(policies) == len(values)):
            raise ValueError("inputs, policies, and values must have the same length")
        self.inputs = inputs
        self.policies = policies
        self.values = values

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], (self.policies[idx], self.values[idx])


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
             array(size * size + 1)
        """
        output = np.zeros(size * size + 1, dtype=np.float32)
        if pos is not None:
            output[pos.x * size + pos.y] = 1.0
        else:
            output[-1] = 1.0
        return output

    @staticmethod
    def encode_value_output(player: GoPlayer, winner: GoPlayer, decay=1.0) -> np.ndarray:
        """
        Returns:
             scalar array in [0, 1]
        """
        value_ = decay if player == winner else 0.0 if winner == GoPlayer.none else -decay
        return np.array(0.5 * (1 + value_), dtype=np.float32)

    def __init__(self, name: str = "alphago_weak_v0", root: str = None, size: int = 19,
                 num_filters: int = 128, device: Optional[Union[str, torch.device]] = None):
        ModelBase.__init__(self, name, root, size)
        self.num_filters = num_filters
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AlphaGoWeakNet(size, self.FEATURES["length"], num_filters).to(self.device)
        if path.exists(self.weights_path):
            self.load()

    def load(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "size": self.size,
            "num_filters": self.num_filters,
        }, self.weights_path)

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
        return (self.FEATURES["length"], self.size, self.size), ((self.size * self.size + 1,), ())

    @staticmethod
    def _preprocess_archive_one(archive: GameArchive, counts: int, idx: int):
        inputs, policies, values = [], [], []
        steps = AlphaGoWeakV0.cached_steps
        cache_dir = archive.cache_dir
        for data_idx in range(idx, len(archive), counts):
            for input_, (policy, value) in AlphaGoWeakV0.game_data_generator(archive[data_idx]):
                inputs.append(input_)
                policies.append(policy)
                values.append(value)
            if len(inputs) >= steps:
                np.save(path.join(cache_dir, f"{idx}.input"), np.array(inputs[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.policy_output"), np.array(policies[:steps], dtype=np.float32))
                np.save(path.join(cache_dir, f"{idx}.value_output"), np.array(values[:steps], dtype=np.float32))
                return True
        return False

    @staticmethod
    def preprocess_archive(archive: GameArchive, counts: int = None):
        if counts is None:
            counts = max(int(len(archive) * 128 / AlphaGoWeakV0.cached_steps), 1)
        if counts <= 0:
            raise ValueError("counts must be greater than zero")
        if path.exists(archive.cache_dir):
            shutil.rmtree(archive.cache_dir)
        makedirs(archive.cache_dir)
        finished_counts = sum(do_cpu_intensive_works(partial(AlphaGoWeakV0._preprocess_archive_one, archive, counts), list(
            range(counts)), total=counts, desc="Preprocess...", use_multiprocessing=True))
        print(f"successfully preprocessed: {finished_counts}/{counts}")

    @staticmethod
    def dataset_from_preprocess(archive: GameArchive):
        length = len(os.listdir(archive.cache_dir)) // 3
        return _PreprocessedGameDataset(archive.cache_dir, AlphaGoWeakV0.cached_steps), length * AlphaGoWeakV0.cached_steps

    def dataset_from_archive(self, archive: GameArchive):
        return _ArchiveGameDataset(archive), int(len(archive) * 128)

    @staticmethod
    def dataset_from_arrays(inputs: Sequence[np.ndarray], policies: Sequence[np.ndarray], values: Sequence[np.ndarray]):
        return _ArrayGameDataset(inputs, policies, values)

    def _batch_to_tensors(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, (policies, values) = batch
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        policies = torch.as_tensor(policies, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device).view(-1)
        return inputs, policies, values

    @staticmethod
    def _policy_loss(predicted: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        return -(expected * torch.log(predicted.clamp_min(1e-7))).sum(dim=1).mean()

    @staticmethod
    def _policy_accuracy(predicted: torch.Tensor, expected: torch.Tensor) -> Tuple[int, int]:
        predicted_idx = torch.argmax(predicted, dim=1)
        expected_idx = torch.argmax(expected, dim=1)
        return int((predicted_idx == expected_idx).sum().item()), int(expected.shape[0])

    def _fit_batches(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                     max_batches: Optional[int] = None, policy_weight: float = 0.7) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, policies, values = self._batch_to_tensors(batch)
            optimizer.zero_grad()
            policy_pred, value_pred = self.model(inputs)
            policy_loss = self._policy_loss(policy_pred, policies)
            value_loss = F.mse_loss(value_pred, values)
            loss = policy_loss * policy_weight + value_loss * (1.0 - policy_weight)
            loss.backward()
            optimizer.step()

            correct, count = self._policy_accuracy(policy_pred.detach(), policies)
            total_correct += correct
            total_count += count
            total_loss += float(loss.item())
            batch_count += 1
        mean_loss = total_loss / batch_count if batch_count else 0.0
        accuracy = total_correct / total_count if total_count else 0.0
        return mean_loss, accuracy

    def fit_from_dataset(self, dataset, steps: int, epochs=100, batch_size=512):
        if steps <= 0:
            raise ValueError("dataset is empty; run download/preprocess first")
        dataloader = DataLoader(dataset, batch_size=batch_size)
        batches_per_epoch = max(steps // batch_size, 1)
        optimizer = torch.optim.Adam(self.model.parameters())
        best_loss = float("inf")
        patience = 5
        stale_epochs = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_correct = 0
            epoch_count = 0

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= batches_per_epoch:
                    break
                inputs, policies, values = self._batch_to_tensors(batch)
                optimizer.zero_grad()
                policy_pred, value_pred = self.model(inputs)
                policy_loss = self._policy_loss(policy_pred, policies)
                value_loss = F.mse_loss(value_pred, values)
                loss = policy_loss * 0.7 + value_loss * 0.3
                loss.backward()
                optimizer.step()

                correct, count = self._policy_accuracy(policy_pred.detach(), policies)
                epoch_loss += float(loss.item())
                epoch_batches += 1
                epoch_correct += correct
                epoch_count += count

            loss = epoch_loss / epoch_batches if epoch_batches else 0.0
            acc = epoch_correct / epoch_count if epoch_count else 0.0
            print(f"epoch {epoch}: Total Loss: {loss:.04f}, Policy Accuracy: {acc:.3%}")
            if loss < best_loss:
                best_loss = loss
                stale_epochs = 0
                self.save()
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    def fit_step_from_dataset(self, dataset) -> Tuple[float, float]:
        dataloader = DataLoader(dataset, batch_size=64)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return self._fit_batches(dataloader, optimizer, policy_weight=0.8)

    def summary(self):
        print(self.model)

    def plot(self, to_file: str = "model.txt"):
        dest_dir = path.dirname(to_file)
        if dest_dir:
            makedirs(dest_dir, exist_ok=True)
        with open(to_file, "w", encoding="utf-8") as f:
            f.write(str(self.model))

    def predict(self, board: GoBoardBase, player: GoPlayer):
        input_ = self.encode_input(player, board)
        input_ = torch.as_tensor(input_[None, ...], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            policy_, value_ = self.model(input_)
        return policy_[0].cpu().numpy(), float(value_[0].cpu().item())

    def state_calculator(self, board: GoBoardBase, player: GoPlayer):
        policy_, value = self.predict(board, player)
        if policy_[-1] > 0.5:
            policy = [], []
        else:
            actions: List[GoPoint] = [pos for pos in board.valid_points(
                player) if board.eye_type(player, pos) < GoEyeType.unknown]
            weights: List[float] = [policy_.item(action.x * self.size + action.y) for action in actions]
            policy = actions, weights
        return policy, value

    @staticmethod
    def policy_evaluator(state, komi=6.5):
        return state[0]

    @staticmethod
    def value_evaluator(state: np.ndarray, komi=6.5):
        return state[1]


AlphaGoWeak = AlphaGoWeakV0
