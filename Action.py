from enum import IntEnum


class Action(IntEnum):
    PUT = 0
    PASS = 1
    BULLSHIT = 2

    @staticmethod
    def num_actions():
        return 3

