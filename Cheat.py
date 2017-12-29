from Player import Player
import numpy as np
from Action import Action
from ActionLogger import ActionLogger

NUM_CARDS = 13


class Cheat:

    def __init__(self, num_players):
        self._players = []
        self._num_players = num_players
        self._pile = []
        self._turn = 0
        self._actions_logger = None
        self._done = False
        self._reward = 0
        self._illegal_reward = -10
        self._win_reward = 10
        self._player_turn = 0

    def reset(self):
        self._players = [Player(i) for i in range(self._num_players)]
        Cheat.deal_cards(self._players)
        self._actions_logger = ActionLogger()
        self._done = False
        self._reward = 0
        self._player_turn = 0
        self._pile = []

    def step(self):

        current_player = self._players[self._turn]
        action, actual_cards, reported_cards = current_player.play(self._actions_logger, len(self._pile))
        print("Player: ", self._turn)
        if action == Action.PASS:
            print("Pass")
            if self.first_put():
                self._reward = self._illegal_reward
            else:
                self._actions_logger.add(self._turn, Action.PASS, None)
                self._turn = (self._turn + 1) % self._num_players

        if action == Action.PUT:
            print("Put: Reported:", str(reported_cards), "Actual: ", str(actual_cards))
            if self.legal_put(reported_cards, actual_cards):
                self._pile.extend(actual_cards)
                current_player.remove_cards(actual_cards)
                self._actions_logger.add(self._turn, Action.PUT, reported_cards)
                self._turn = (self._turn + 1) % self._num_players

                if current_player.finished():
                    if np.array_equal(reported_cards, actual_cards):
                        self._reward = self._win_reward
                        self._done = True
                    else:
                        current_player.recieve_cards(self._pile)
                        self._pile = []
            else:
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
                self._reward = self._illegal_reward

        return self._done

    def done(self):
        return self._done

    def first_put(self):
        return len(self._pile) == 0

    @staticmethod
    def deal_cards(players):
        num_players = len(players)
        card_deck = np.random.permutation(np.repeat(range(NUM_CARDS), 4, axis=0))
        num_cards = int(np.floor(len(card_deck) / num_players))
        for i in range(num_players):
            players[i].recieve_cards(card_deck[i * num_cards:(i + 1) * num_cards])

    @staticmethod
    def all_reported_cards_of_same_type(reported_cards):
        return all([reported_cards[i] == reported_cards[0] for i in range(len(reported_cards))])

    @staticmethod
    def num_reported_equals_actual(reported_cards, actual_cards):
        return len(reported_cards) == len(actual_cards)

    def legal_put(self, reported_cards, actual_cards):
        if self.first_put():
            if len(reported_cards) == len(actual_cards):
                return True
        else:
            current_hand = self._actions_logger.get_current_hand()
            if Cheat.all_reported_cards_of_same_type(reported_cards) \
                    and Cheat.num_reported_equals_actual(reported_cards, actual_cards) \
                    and (reported_cards[0] == current_hand):
                return True

        return False