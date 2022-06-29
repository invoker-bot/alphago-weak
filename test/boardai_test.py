import unittest
import numpy as np
from alphago_weak.board import *
from alphago_weak.model.alpha_go_weak import GoBoardAIV0


class TestGoBoardAI(unittest.TestCase):
    def test_v0(self):
        b = GoBoardAIV0()
        inputs = np.zeros_like(b.inputs)
        self.assertTrue(np.all(inputs == b.inputs))
        inputs[0] = 1
        inputs[1, 3, 4] = 1
        b.play(GoPoint(3, 4), player=GoPlayer.black)
        print(inputs)
        print(b.inputs)
        self.assertTrue(np.all(inputs == b.inputs))


if __name__ == '__main__':
    unittest.main()
