from Action import Action
import numpy as np
import cheat_utils
from PlayerNetwork import PlayerNetwork


class Player:
    def __init__(self, player_id):
        self._id = player_id
        self._hand = []
        self._s_size = 18
        self._network = PlayerNetwork(1e-3, self._s_size, 3, 128, player_id)
        self._ep_hist = []
        self._previous_action = 0
        self._previous_state = [0] * self._s_size
        self._plays_count = 0
        self._immediate_reward = 0

    def reset(self):
        self._ep_hist = []
        self._previous_action = 0
        self._previous_state = [0] * self._s_size
        self._plays_count = 0
        self._immediate_reward = 0
        self._hand = []

    def recieve_cards(self, cards):
        self._hand.extend(cards)
        self._hand.sort()

    def num_cards_hand(self):
        return len(self._hand)

    def id(self):
        return self._id

    def play(self, sess, game_history, current_hand, pile_size, players_hands):

        self._plays_count += 1

        current_state = cheat_utils.encode_state(game_history, current_hand, pile_size, players_hands, self._hand)
        self._ep_hist.append([self._previous_state, self._previous_action, self._immediate_reward, current_state])

        action_dist = sess.run(self._network.action_distribution, feed_dict={self._network.state_in: [current_state]})

        selected_action = cheat_utils.dist_selection(action_dist[0])
        self._previous_action = selected_action
        self._previous_state = current_state

        if selected_action == Action.PUT:
            num_drop = min(len(self._hand), 4)
            actual_cards = self._hand[-num_drop:]
            if current_hand is not None:
                reported_cards = np.repeat(current_hand, num_drop)
            else:
                reported_cards = np.repeat(actual_cards[-1], num_drop)
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

    def update_reward(self, r):
        self._immediate_reward = r

    def get_history(self):
        look_back = min(50, len(self._ep_hist))
        return self._ep_hist[-look_back:]

    def network(self):
        return self._network