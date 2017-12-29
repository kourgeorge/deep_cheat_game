from Action import Action
import numpy as np
import cheat_utils
from PlayerNetwork import PlayerNetwork


class Player:
    def __init__(self, player_id):
        self._id = player_id
        self._hand = []
        self._s_size = 19
        self._network = PlayerNetwork(1e-3, self._s_size, 3, 128, player_id)

    def recieve_cards(self, cards):
        self._hand.extend(cards)
        self._hand.sort()
        self.playes_count = 0

    def num_cards_hand(self):
        return len(self._hand)

    def id(self):
        return self._id

    def play(self, sess, game_history, current_hand, pile_size, players_hands):
        self.playes_count += 1

        state = cheat_utils.encode_state(game_history, current_hand, pile_size, players_hands, self._hand)
        action_dist = sess.run(self._network.action_distribution, feed_dict={self._network.state_in: [state]})

        selected_action = cheat_utils.dist_selection(action_dist[0])

        # selected_action = np.random.randint(low=0, high=Action.num_actions())
        # selected_action = Action.PUT

        if selected_action == Action.PUT:
            num_drop = min(len(self._hand), 4)
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

    def hand_size(self):
        return len(self._hand)
