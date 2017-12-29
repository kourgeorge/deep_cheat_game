from Action import Action
import numpy as np


class Player:

    def __init__(self, id):
        self._id = id
        self._hand = []

    def recieve_cards(self, cards):
        self._hand.extend(cards)
        self._hand.sort()

    def num_cards_hand(self):
        return len(self._hand)

    def id(self):
        return self._id

    def play(self, action_logger, pile_size):

        selected_action = np.random.randint(low=0, high=Action.num_actions())
        #if pile_size == 0:
        #selected_action = Action.PUT

        if selected_action == Action.PUT:
            current_hand = action_logger.get_current_hand()
            num_drop=min(len(self._hand), 4)
            actual_cards = self._hand[-num_drop:]
            if current_hand is not None:
                reported_cards = np.repeat(current_hand, num_drop)
            else:
                reported_cards = np.repeat(actual_cards[-1], 4)
            return Action.PUT, actual_cards, reported_cards

        if selected_action == Action.PASS:
            return Action.PASS, None, None

        if selected_action == Action.BULLSHIT:
            return Action.BULLSHIT, None, None


    def finished(self):
        return len(self._hand) == 0

    def remove_cards(self, cards):
        for card in cards:
            self._hand.remove(card)