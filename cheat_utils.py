import numpy as np
import Config as config


def deal_cards(players, num_cards):
    num_players = len(players)
    card_deck = np.random.permutation(np.repeat(range(num_cards), config.num_cards_in_rank, axis=0))
    num_cards = int(np.floor(len(card_deck) / num_players))
    for i in range(num_players):
        players[i].receive_cards(card_deck[i * num_cards:(i + 1) * num_cards])


def all_reported_cards_of_same_type(reported_cards):
    return all([reported_cards[i] == reported_cards[0] for i in range(len(reported_cards))])


def num_reported_equals_actual(reported_cards, actual_cards):
    return len(reported_cards) == len(actual_cards)


def extract_episode_history(ep_history):
    episode_states = [inter_data[0] for inter_data in ep_history]
    episode_actions = [inter_data[1] for inter_data in ep_history]
    episode_played_cards = [inter_data[2] for inter_data in ep_history]
    episode_rewards = np.array([inter_data[3] for inter_data in ep_history])
    episode_length = len(episode_rewards)

    return episode_states, episode_actions, episode_played_cards, episode_rewards, episode_length

def rotate(l, n):
    return l[n:] + l[:n]


def legal_put(first_put, current_rank, reported_cards, actual_cards):
    if first_put:
        if len(reported_cards) == len(actual_cards):
            return True
    else:
        if len(reported_cards) >0 \
                and all_reported_cards_of_same_type(reported_cards) \
                and num_reported_equals_actual(reported_cards, actual_cards) \
                and (reported_cards[0] == current_rank):
            return True

    return False


def encode_state(game_history, current_rank, pile_size, palyers_hands, own_cards):
    if current_rank is None:
        current_rank = -1
    return [current_rank] + [pile_size] + palyers_hands + encode_hand_cards(own_cards)


def encode_hand_cards(hands_cards):
    encoding = [hands_cards.count(i) for i in range(config.num_ranks)]

    return encoding


def decode_hand_cards(decoded_hand_cards):
    cards = []
    for i in range(len(decoded_hand_cards)):
        cards.extend(decoded_hand_cards[i] * [i])
    return cards

def dist_selection(dist):
    select_prob = np.random.choice(dist, p=dist)
    selection = np.argmax(dist == select_prob)

    return selection


def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
