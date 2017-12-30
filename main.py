from Player import Player
from Cheat import Cheat
import tensorflow as tf
import Config as config
from PlayerTrainer import PlayerTrainer
import cheat_utils
import numpy as np


def main():
    players = [Player(i) for i in range(config.num_players)]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        env = Cheat(players)
        trainers = [PlayerTrainer(players[i].network(), sess) for i in range(config.num_players)]

        illegal_count = []
        wins = []
        rounds = []
        for ep in range(config.episodes):
            done, turn, pile_size, players_hands, current_hand, game_history, _, _ = env.reset()
            plays=0

            illegal_c = 0
            while not done:
                plays += 1
                player = players[turn]

                action, actual_cards, reported_cards = player.play(sess, game_history, current_hand, pile_size,
                                                                   players_hands)
                done, turn, pile_size, players_hands, current_hand, game_history, illegal_action, reward = env.step(
                    action, actual_cards, reported_cards)

                player.update_reward(reward)

                if illegal_action:
                    illegal_c +=1
                    print("illegal")

                if (plays % 500 == 0 and plays != 0) or done:
                    for i in range(config.num_players):
                        ep_history = players[i].get_history()
                        episode_states, episode_actions, episode_rewards, episode_length = \
                            cheat_utils.extract_episode_history(ep_history)

                        discounted_episode_rewards = cheat_utils.discount_rewards(episode_rewards, config.gamma)

                        trainers[i].accumulate_action_gradients(episode_states, episode_actions, discounted_episode_rewards)

                        # if i % update_frequency == 0 and i != 0:
                        trainers[i].update_action_model()

                if done:
                    winner = player._id
                    game_wins = [0, 0, 0, 0]
                    game_wins[winner] = 1
                    wins.append(game_wins)
                    illegal_count.append(illegal_c)
                    rounds.append(plays)

            if ep % 50 == 0 and ep > 0:
                print("Illegal moves: ", np.mean(illegal_count[-50:]),". Rounds: ", np.mean(rounds[-50:]), ". Stats: ", np.sum(wins[-50:], axis=0))
                x=1


if __name__ == '__main__':
    main()
