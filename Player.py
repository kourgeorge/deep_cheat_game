from Action import Action
import numpy as np
import cheat_utils
import Config as config
from PlayerNetwork import PlayerNetwork
import math


class Player:
    def __init__(self, player_id, player_network):
        self._id = player_id
        self._hand = []
        self._s_size = 18
        self._network = player_network
        self._ep_hist = []
        self._previous_action = 0
        self._previous_played_cards = [0]*config.num_ranks
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
        self._previous_played_cards = [0]*config.num_ranks

    def receive_cards(self, cards):
        self._hand.extend(cards)
        self._hand.sort()

    def num_cards_hand(self):
        return len(self._hand)

    def id(self):
        return self._id

    def play(self, sess, game_history, current_rank, pile_size, players_hands):

        self._plays_count += 1

        current_state = cheat_utils.encode_state(game_history, current_rank, pile_size, players_hands, self._hand)

        self._ep_hist.append([self._previous_state, self._previous_action, self._previous_played_cards, self._immediate_reward, current_state])

        action_dist = sess.run(self._network.action_distribution, feed_dict={self._network.state_in: [current_state]})
        suggested_hand = sess.run(self._network.cards_selection, feed_dict={self._network.state_in: [current_state]})

        selected_cards_enc = self.filter_available(suggested_hand[0], self._hand)
        selected_cards = cheat_utils.decode_hand_cards(selected_cards_enc)

        selected_action = cheat_utils.dist_selection(action_dist[0])
        self._previous_action = selected_action
        self._previous_state = current_state

        if selected_action == Action.PUT and len(selected_cards)>0:
            actual_cards = selected_cards
            if current_rank is not None:
                reported_cards = np.repeat(current_rank, len(selected_cards))
            else:
                reported_cards = np.repeat(actual_cards[-1], len(selected_cards))

            self._previous_played_cards = selected_cards_enc
            return Action.PUT, actual_cards, reported_cards

        if selected_action == Action.BULLSHIT:
            self._previous_played_cards = [0]*config.num_ranks
            return Action.BULLSHIT, None, None

        self._previous_played_cards = [0]*config.num_ranks
        return Action.PASS, None, None

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

    def filter_available(self, suggested_hand, current_hand):
        current_hand = cheat_utils.encode_hand_cards(current_hand)
        selected_hand = [math.floor(min(suggested_hand[rank]*4, current_hand[rank])) for rank in range(config.num_ranks)]
        return selected_hand