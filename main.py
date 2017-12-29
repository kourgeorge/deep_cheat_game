from Player import Player
from Cheat import Cheat


def main():
    num_players = 2

    env = Cheat(num_players)
    env.reset()
    done = False
    while not done:
        done = env.step()



if __name__ == '__main__':
    main()