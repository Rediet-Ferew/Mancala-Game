import copy
# from Players import QLearningAgent, SARSAAgent, DQNAgent

class Board:
    def __init__(self):
        """Initializes the Board."""
        self.reset()

    def reset(self):
        """Resets all the variants of the Board."""
        self.PITS = 6
        self.PLAYER1_PITS = [4] * self.PITS
        self.PLAYER2_PITS = [4] * self.PITS
        self.bankPits = [0, 0]
    
    def copy(self):
        """Copies the object and returns the copy."""
        new_board = Board()
        new_board.PITS = self.PITS
        new_board.PLAYER1_PITS = [pit for pit in self.PLAYER1_PITS]
        new_board.PLAYER2_PITS = [pit for pit in self.PLAYER2_PITS]
        new_board.bankPits = [self.bankPits[0], self.bankPits[1]]
        return new_board

    def __repr__(self):
        """Custom repr() method to represent this object."""
        ret = "P L A Y E R  2\n"
        ret += "------------------------------------------------------------\n"
        ret += str(self.bankPits[1]) + "\t"
        for elem in range(len(self.PLAYER2_PITS) - 1, -1, -1):
            ret += str(self.PLAYER2_PITS[elem]) + "\t"
        ret += "\n\t"
        for elem in self.PLAYER1_PITS:
            ret += str(elem) + "\t"
        ret += str(self.bankPits[0])
        ret += "\n------------------------------------------------------------"
        ret += "P L A Y E R  1\n"
        return ret

    # def is_legal_move(self, player, pit):
    #     """Return the legality of the proposed move 'pit' by the player 'player'."""
    #     if player.num == 1:
    #         pits = self.PLAYER1_PITS
    #     else:
    #         pits = self.PLAYER2_PITS
    #     return 0 < pit <= len(pits) and pits[pit-1] != 0

    def is_game_over(self):
        """Returns True if the game has ended."""
        end = True
        for j in range(len(self.PLAYER1_PITS)):
            if self.PLAYER1_PITS[j] != 0:
                end = False
        if end:
            return True
        for k in range(len(self.PLAYER2_PITS)):
            if self.PLAYER2_PITS[k] != 0:
                end = False
        if end:
            return True

    def has_won(self, player):
        """Returns 'True' if the player 'player' wons the game or not."""
        if self.is_game_over():
            opp_player = 2 - player + 1
            return self.bankPits[player -1] > self.bankPits[opp_player - 1]
        else:
            return False

    # def get_possible_moves(self, player):
    #     """Return the list of legal moves in accordance with the rules of the game."""
    #     if player.num == 1:
    #         pits = self.PLAYER1_PITS
    #     else:
    #         pits = self.PLAYER2_PITS
    #     possible_moves = []
    #     # book non-zero pits (possible moves)
    #     for i in range(len(pits)):
    #         if pits[i] != 0:
    #             possible_moves.append(i+1)
    #     return possible_moves

    def move_helper(self, player, pit):
        """Helper function for the 'make_move' method, returns a boolean for repeating a turn."""
        if player.num == 1:
            pits = self.PLAYER1_PITS
            opp_pits = self.PLAYER2_PITS
        else:
            pits = self.PLAYER2_PITS
            opp_pits = self.PLAYER1_PITS
        init_pits = pits
        stones = pits[pit-1]
        pits[pit-1] = 0
        pit += 1
        repeat_turn = False
        while stones > 0:
            repeat_turn = False
            # seeding stones into pits
            while pit <= len(pits) and stones > 0:
                pits[pit-1] += 1
                stones -= 1
                pit += 1
            if stones == 0:
                break
            if pits == init_pits: # ??
                self.bankPits[player.num - 1] += 1
                stones -= 1
                repeat_turn = True
            pits, opp_pits = opp_pits, pits
            pit = 1
        if repeat_turn:
            return True

        # capturing the other side's stones
        if pits == init_pits and pits[pit - 2] == 1:
            self.bankPits[player.num - 1] += opp_pits[(self.PITS - pit) + 1]
            opp_pits[(self.PITS - pit) + 1] = 0
            pits[pit - 2] = 0
        return False
    

    def get_score(self, player):
        """Return the running score of the player 'player'."""
        return self.bankPits[player.num - 1]

    def make_move(self, player, pit):
        """Makes the actual move in the game - returns whether the game can continue or not."""
        repeat = self.move_helper(player, pit)
        if self.is_game_over(): # If the game has ended
            for i in range(len(self.PLAYER1_PITS)):
                self.bankPits[0] += self.PLAYER1_PITS[i]
                self.PLAYER1_PITS[i] = 0
            for j in range(len(self.PLAYER2_PITS)):
                self.bankPits[1] += self.PLAYER2_PITS[j]
                self.PLAYER2_PITS[j] = 0
            return False
        else:
            return repeat

    def future_lookup(self, curr_player, pit):
        """Helper for looking up the game in the future, returns the newly created game state."""
        if curr_player.num == 1:
            pits = self.PLAYER1_PITS
            opp_pits = self.PLAYER2_PITS
        else:
            pits = self.PLAYER2_PITS
            opp_pits = self.PLAYER1_PITS
        init_pits = pits
        stones = pits[pit-1]
        pits[pit-1] = 0
        pit += 1
        repeat_turn = False
        while stones > 0:
            repeat_turn = False
            # seeding stones into pits
            while pit <= len(pits) and stones > 0:
                pits[pit-1] += 1
                stones -= 1
                pit += 1
            if stones == 0:
                break
            if pits == init_pits:
                self.bankPits[curr_player.num - 1] += 1
                stones -= 1
                repeat_turn = True
            pits, opp_pits = opp_pits, pits
            pit = 1
        if repeat_turn:
            return self, True

        if pits == init_pits and pits[pit - 2] == 1:
            self.bankPits[curr_player.num - 1] += opp_pits[(self.PITS - pit) + 1]
            opp_pits[(self.PITS - pit) + 1] = 0
            pits[pit - 2] = 0
        return self, False
    
    def miniMaxMove(self, player, depth, max_for_player, other_player):
        """Maximizes the minimum score possible by playing a move, returns tuple of best move."""
        # When we can't make any more moves, just calculate the "final score" of the board.
        if depth == 0 or self.is_game_over():
            return None, self.get_score(max_for_player)

        # Get our list of possible moves and set a default we'll definitely beat.
        moves = self.get_possible_moves(player)
        # print(f"moves => {moves}")
        if not moves:
            return None, self.get_score(max_for_player)
        maximise = max_for_player == player
        worst_score = float('-inf') if maximise else float('inf')
        best_move = moves[0], worst_score
        # print(f"best move => {best_move[0]}")

        for move in moves:
            new_board, is_repeat = self.future_lookup(player, move)
            # print(new_board)
            # if not is_repeat: 
            #     player, other_player = other_player, player
            # recursive call for depth times
            _move, score = new_board.miniMaxMove(other_player, depth - 1, max_for_player, player)

            set_new_max = maximise and score >= best_move[1]
            set_new_min = (not maximise) and score <= best_move[1]
            if _move and (set_new_max or set_new_min):
                best_move = move, score
        return best_move

    # def host_game(self, player1, player2):
    #     """Hosts the game."""
    #     self.reset()
    #     curr_player = player1
    #     wait_player = player2
    #     # main loop of the game - continues until it ends
    #     while not (self.is_game_over()):
    #         again = True
    #         while again:
    #             print(self)
    #             move = curr_player.choose_move(self, wait_player)
    #             # validate the chosen move
    #             while not (self.is_legal_move(curr_player, move)):
    #                 print(f"{move} is not legal")
    #                 move = curr_player.choose_move(self, wait_player)
    #             again = self.make_move(curr_player, move)
    #         curr_player, wait_player = wait_player, curr_player
    #     # End message
    #     print(self)
    #     if self.has_won(curr_player.num):
    #         print(f"Player {curr_player} wins with score {self.get_score(curr_player)}")

    #     elif self.has_won(wait_player.num):
    #         print(f"Player {wait_player} wins with score {self.get_score(curr_player)}")
    #     else:
    #         print("Tie Game")
    # def get_possible_moves(self, player):
    #     """Returns a list of all possible moves for the given player."""
    #     possible_moves = []
    #     for pit in range(1, 7):
    #         if self.is_legal_move(player, pit):
    #             possible_moves.append(pit)
    #     return possible_moves
    def get_possible_moves(self, player):
        """Returns a list of all possible moves for the given player."""
        if player.num == 1:
            return [pit for pit in range(1, 7) if self.is_legal_move(player, pit)]
        elif player.num == 2:
            return [pit for pit in range(8, 14) if self.is_legal_move(player, pit)]
        else:
            return []

    def is_legal_move(self, player, pit):
        """Checks if the move is legal for the given player and pit."""
        if player.num == 1:
            return self.PLAYER1_PITS[pit - 1] > 0
        elif player.num == 2:
            return self.PLAYER2_PITS[pit - 8] > 0
        else:
            return False
    def host_game(self, player1, player2):
        """Hosts a game between two players."""
        self.display()
        while not self.is_game_over():
            curr_player = player1 if self.current_player == player1.num else player2
            wait_player = player2 if curr_player == player1 else player1
            move = curr_player.choose_move(self, wait_player)
            self.make_move(curr_player, move)
            self.display()
            self.switch_player()
        self.display_winner()
    
    def play_game(self, player1, player2, max_rounds=1000):
        for _ in range(max_rounds):
            self.reset()
            curr_player = player1
            wait_player = player2
            while not self.is_game_over():
                state = curr_player.get_state(self)
                move = curr_player.choose_move(self, wait_player)
                while not self.is_legal_move(curr_player, move):
                    move = curr_player.choose_move(self, wait_player)
                reward = self.make_move(curr_player, move)
                next_state = curr_player.get_state(self)
                done = self.is_game_over()
                curr_player.update_q_value(state, move, reward, next_state, done)
                curr_player, wait_player = wait_player, curr_player
