from Player import Player
from Cheat import Cheat
import tensorflow as tf
import Config as config


def main():

    players = [Player(i) for i in range(config.num_players)]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        env = Cheat(players)
        done, turn, pile_size, players_hands, current_hand, game_history, _, _ = env.reset()

        while not done:
            player = players[turn]
            action, actual_cards, reported_cards = player.play(sess, game_history, current_hand, pile_size, players_hands)
            done, turn, pile_size, players_hands, current_hand, game_history, illegal_action, reward = env.step(action, actual_cards, reported_cards)



            if illegal_action:
                print("illegal")


if __name__ == '__main__':
    main()