import numpy as np
from Action import Action
from ActionLogger import ActionLogger
import cheat_utils

NUM_CARDS = 13


class Cheat:

    def __init__(self, players):
        self._players = players
        self._num_players = len(players)
        self._pile = []
        self._turn = 0
        self._actions_logger = None
        self._done = False
        self._player_turn = 0

        self._illegal_action = False
        self._reward = 0
        self._illegal_reward = -3
        self._max_reward = 3

    def reset(self):
        cheat_utils.deal_cards(self._players, NUM_CARDS)
        self._actions_logger = ActionLogger()
        self._done = False
        self._player_turn = 0
        self._pile = []
        self._illegal_action = False
        self._reward = 0

        return self.state()

    def step(self, action, actual_cards, reported_cards):

        current_player = self._players[self._turn]
        print("Player: ", self._turn, " Hand:", current_player._hand)

        self._illegal_action = False
        self._reward = 0

        if action == Action.PASS:
            print("Pass")
            if self.first_put():
                self._illegal_action = True
                self._reward = self._illegal_reward
            else:
                self._actions_logger.add(self._turn, Action.PASS, None)
                self._turn = (self._turn + 1) % self._num_players

        if action == Action.PUT:
            print("Put: Reported:", str(reported_cards), "Actual: ", str(actual_cards))
            if cheat_utils.legal_put(self.first_put(), self.current_hand(), reported_cards, actual_cards):
                self._pile.extend(actual_cards)
                current_player.remove_cards(actual_cards)
                self._actions_logger.add(self._turn, Action.PUT, reported_cards)
                self._turn = (self._turn + 1) % self._num_players

                if current_player.finished():
                    if np.array_equal(reported_cards, actual_cards):
                        self._done = True
                        self._reward = self._max_reward
                    else:
                        current_player.recieve_cards(self._pile)
                        self._pile = []
            else:
                self._illegal_action = True
                self._reward = self._illegal_reward

        if action == Action.BULLSHIT:
            print("Bullshit")
            if not self.first_put():
                cards, put_player_id = self._actions_logger.get_last_put_info()
                num_put = len(cards)
                self._actions_logger.add(current_player, Action.BULLSHIT, None)
                if np.array_equal(cards, self._pile[-num_put:]):
                    current_player.recieve_cards(self._pile)
                    self._pile = []
                    self._turn = put_player_id
                else:
                    self._players[put_player_id].recieve_cards(self._pile)
                    self._pile = []
                    self._turn = self._turn
            else:
                self._illegal_action = True
                self._reward = self._illegal_reward

        return self.state()

    def done(self):
        return self._done

    def first_put(self):
        return len(self._pile) == 0

    def current_hand(self):
        return self._actions_logger.get_current_hand()

    def state(self):
        players_hands = cheat_utils.rotate([player.hand_size() for player in self._players], self._turn)
        return self._done, self._turn, len(
            self._pile), players_hands, self._actions_logger.get_current_hand(), \
               self._actions_logger.get_game_history(), self._illegal_action, self._reward

