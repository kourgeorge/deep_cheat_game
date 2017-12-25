import unittest
from Cheat import Cheat


class TestCheat(unittest.TestCase):

    def test_init(self):
        env = Cheat(num_players=2)
        env.reset()

        for i in range(100):
            env.play_turn()
