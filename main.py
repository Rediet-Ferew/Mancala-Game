# from Board import *
# from Players import *

# def main():

#     """The main entry of the game."""   
#     player1 = QLearningAgent(num=1)
#     player2 = QLearningAgent(num=2)
#     board = Board()
#     board.play_game(player1, player2)

#     # To play a game and see the result:
#     board.host_game(player1, player2)

# if __name__ == "__main__":
#     main()
from Board import Board
from Agents import QLearningAgent, SARSAAgent, DQNAgent, Agent
from Players import Player


def main():
    
    player1 = QLearningAgent(1)
    player2 = SARSAAgent(2)
    player3 = DQNAgent(1, 14, 6)
    player4 = Player(player_num=1, player_type=Player.RANDOM)
    cnt_14 = 0
    cnt_24 = 0
    cnt_34 = 0

    

    board = Board()
    print("Starting QLearning vs Random")
    board.host_game(player1, player4)
        
        
        

    board = Board()  # Reset board for the next game
    print("Starting DQN vs Random")
    board.host_game(player3, player4)

    board = Board()  # Reset board for the next game
    print("Starting SARSA vs Random")
    board.host_game(player2, player4)

if __name__ == "__main__":
    main()
