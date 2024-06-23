import random
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from Board import Board

class Agent:
    HUMAN = 0
    RANDOM = 1
    AI = 2

    def __init__(self, player_num, player_type, ply=0):
        """Initializes the players."""
        self.AI_DEPTH = 4
        self.num = player_num
        self.opp = 2 - player_num + 1
        self.type = player_type
        self.ply = ply

    def __repr__(self):
        """Represents this object - returns the player's number."""
        return str(self.num)

    def score(self, board):
        if board.has_won(self.num):
            return 1
        elif board.has_won(self.opp):
            return -1
        else:
            return 0

    def choose_move(self, board, other_player):
        """Accepts moves from a 'HUMAN' player or else generates random moves if it is a 'RANDOM' player or else returns an AI generated move."""
        if self.type == self.HUMAN:
            while True:
                try:
                    move = int(input("Enter Your Move: "))
                except ValueError:
                    print("Please Enter a valid move")
                else:
                    break

            while not board.is_legal_move(self, move):
                print("Your move is not valid")
                break
            return move
        # random generation AI
        if self.type == self.RANDOM:
            move = random.choice(board.get_possible_moves(self))
            print(f'random move => {move}')
            return move
        # AI opponent with miniMax
        if self.type == self.AI:
            new_board = board.copy()
            move, _ = new_board.miniMaxMove(self, self.AI_DEPTH, self, other_player)
            print(f"AI(miniMax) move => {move}")
            return move

class QLearningAgent(Agent):
    def __init__(self, num, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        super().__init__(num, Agent.AI)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self, board):
        """Convert board state to a tuple (immutable) to use as keys in Q-table."""
        return (tuple(board.PLAYER1_PITS), tuple(board.PLAYER2_PITS), tuple(board.bankPits))

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_move(self, board, opponent):
        """Choose an action using epsilon-greedy strategy."""
        
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random move
            possible_moves = board.get_possible_moves(self)
            return random.choice(possible_moves)
        else:
            # Exploitation: Choose the move with the highest Q-value
            state = self.get_state(board)
            possible_moves = board.get_possible_moves(self)
            q_values = [self.get_q_value(state, move) for move in possible_moves]
            
            max_q_value = max(q_values)
            best_moves = [move for move, q in zip(possible_moves, q_values) if q == max_q_value]
            return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning update rule."""
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        current_q_value = self.get_q_value(state, action)
        future_q_values = [self.get_q_value(next_state, a) for a in range(1, 7)]
        max_future_q_value = max(future_q_values)
        new_q_value = (current_q_value + 
                       self.learning_rate * (reward + self.discount_factor * max_future_q_value - current_q_value))
        self.q_table[(state, action)] = new_q_value

# SARSAAgent Implementation
class SARSAAgent(Agent):
    def __init__(self, num, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        super().__init__(num, Agent.AI)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self, board):
        """Convert board state to a tuple (immutable) to use as keys in Q-table."""
        return (tuple(board.PLAYER1_PITS), tuple(board.PLAYER2_PITS), tuple(board.bankPits))

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_move(self, board, opponent):
        """Choose an action using epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random move
            possible_moves = board.get_possible_moves(self)
            if not possible_moves:
                print("No possible moves available.")
                return None  # or any other suitable handling for this case
            return random.choice(possible_moves)
        else:
            # Exploitation: Choose the move with the highest Q-value
            state = self.get_state(board)
            possible_moves = board.get_possible_moves(self)
            q_values = [self.get_q_value(state, move) for move in possible_moves]
            
            # Check if q_values is empty
            if not any(q_values):
                print("No Q-values available. Choosing randomly.")
                return random.choice(possible_moves)
            
            max_q_value = max(q_values)
            best_moves = [move for move, q in zip(possible_moves, q_values) if q == max_q_value]
            return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state, next_action):
        """Update Q-value using the SARSA update rule."""
        current_q_value = self.get_q_value(state, action)
        next_q_value = self.get_q_value(next_state, next_action)
        new_q_value = (current_q_value + 
                       self.learning_rate * (reward + self.discount_factor * next_q_value - current_q_value))
        self.q_table[(state, action)] = new_q_value



class DQNAgent(Agent):
    def __init__(self, num, state_size, action_size, learning_rate=0.001, epsilon=0.1, discount_factor=0.9):
        super().__init__(num, Agent.AI)
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def get_state(self, board):
        return np.array(board.PLAYER1_PITS + board.PLAYER2_PITS + board.bankPits).reshape(1, -1)

    def choose_move(self, board, opponent):
        possible_moves = board.get_possible_moves(self)
        if not possible_moves:
            print("No possible moves available.")
            return None  # or any other suitable handling for this case
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_moves)
        state = self.get_state(board)
        q_values = self.model.predict(state)
        q_values = [q_values[0][move-1] for move in possible_moves]
        max_q_value = max(q_values)
        best_moves = [move for move, q in zip(possible_moves, q_values) if q == max_q_value]
        return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state):
        target = reward
        if next_state is not None:
            next_q_values = self.model.predict(next_state)
            target = reward + self.discount_factor * np.max(next_q_values[0])
        target_f = self.model.predict(state)
        target_f[0][action-1] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
