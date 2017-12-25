from Player import Player
import numpy as np
from Action import Action
from ActionLogger import ActionLogger

NUM_CARDS = 13


class Cheat:

    def __init__(self, num_players):
        self._num_players = num_players
        self._players = []
        self._pile = []
        self._turn = 0
        self._actions_logger = None
        self._done = False

    def reset(self):
        self._players = [Player() for i in range(self._num_players)]
        card_deck = np.random.permutation(np.repeat(range(NUM_CARDS), 4, axis=0))
        num_cards = int(np.floor(len(card_deck) / self._num_players))
        for i in range(self._num_players):
            self._players[i].get_cards(card_deck[i * num_cards:(i + 1) * num_cards])

        self._done = False
        self._pile = []
        self._turn = 0
        self._actions_logger = ActionLogger()

    def play_turn(self):

        current_player = self._players[self._turn]
        action, actual_cards, reported_cards = current_player.play(self._actions_logger, len(self._pile))

        if action == Action.PASS:
            self._actions_logger.add(self._turn, Action.PASS, None)
            self._turn = (self._turn + 1) % self._num_players

        if action == Action.PUT:

            if any(reported_cards[i]!=reported_cards[0] for i in range(reported_cards)):
                return

            self._pile.append(actual_cards)
            self._actions_logger.add(self._turn, Action.PUT, reported_cards)
            self._turn = (self._turn + 1) % self._num_players

            if current_player.finished():
                if np.array_equal(reported_cards,actual_cards):
                    self._done = True
                else:
                    current_player.get_cards(self._pile)
                    self._pile = []
                    self._actions_logger.add(self._turn, Action.PUT, reported_cards)

        if action == Action.BULLSHIT:
            put_player_id, _, put_reported_cards = self._actions_logger.get_last_put_action()
            num_put = len(put_reported_cards)
            self._actions_logger.add(current_player, Action.BULLSHIT, None)
            if np.array_equal(put_reported_cards, self._pile[-num_put:]):
                current_player.get_cards(self._pile)
                self._pile = []
                self._turn = put_player_id
            else:
                self._players[put_player_id].get_cards(self._pile)
                self._pile = []
                self._turn = self._turn

        return self._done