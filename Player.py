from Action import Action
import numpy as np

class Player:

    def __init__(self):
        self._hand = []

    def get_cards(self, cards):
        self._hand = self._hand.append(cards)

    def num_cards_hand(self):
        return len(self._hand)

    def play(self, action_logger, pile_size):

        selected_action = np.random.randint(low=0,high=Action.num_actions())

        if selected_action==Action.PUT:
            cards


        return Action.PASS, None, None

