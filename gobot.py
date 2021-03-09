from goboard import GoBoard, GoPlayer, GoPoint
import numpy as np
from gotypes import GoString
from typing import *


class GoBoardAI(GoBoard):
    Features = 9
    """
    Feature name            num of planes   Description
    Stone colour            3               black / white / next player
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
        shape = self._grid.shape
        input_ = np.zeros((self.Features, shape[0], shape[1]), dtype=dtype)

        for pos in self.__iter__():
            player = self.__getitem__(pos)
            x, y = pos
            if player is GoPlayer.black:
                input_.itemset((self.Offset["stone_color_black"], x, y), 1)
            elif player is GoPlayer.black:
                input_.itemset((self.Offset["stone_color_white"], x, y), 1)

            if player is self._next_player:
                input_.itemset((self.Offset["next_player_color"], x, y), 1)

            if self.is_point_a_fake_eye(pos):
                input_.itemset((self.Offset["sensibleness"], x, y), 1)
            # if self.is_valid_point(pos):
            #    input_.itemset((self.Offset["valid_position"], x, y), 1)
        strings = self.get_strings()
        input_[self.Offset["valid_position"]] = self.valid_points(strings, dtype)
        for string in strings:
            liberties = len(string.liberties)
            if liberties == 1:
                input_.itemset((self.Offset["liberties"], x, y), 1)
            elif liberties == 2:
                input_.itemset((self.Offset["liberties"] + 1, x, y), 1)
            elif liberties == 3:
                input_.itemset((self.Offset["liberties"] + 2, x, y), 1)
            else:
                input_.itemset((self.Offset["liberties"] + 3, x, y), 1)
        return input_

    def encode_policy_output(self, pos: GoPoint, dtype=np.float16) -> np.ndarray:
        output = np.zeros(self._grid.shape, dtype=dtype)
        output.itemset(pos, 1)
        return output

    def encode_value_output(self, player: GoPlayer, dtype=np.float16) -> np.ndarray:
        value = 1 if player is GoPlayer.black else 0 if player is GoPlayer.none else -1
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
