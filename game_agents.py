from game_states import *
from tetris_game import Controls, Agent, gravity_to_frames
from util import Adj_Grid_Names, lookup
from blocks import turn
from time import time
import numpy as np
from random import uniform
from math import floor

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = lookup(evalFn, globals())
        self.depth = int(depth)

def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.get_score()

def rowDepthEvaluationFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    score = currentGameState.get_score()
    comboCount = currentGameState.get_combo_count()
    queue = currentGameState.get_queue()
    hold = currentGameState.get_hold()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    hard_drop_loc, _ = piece.hard_drop(board, piece_loc)

    board_list = board.asList(piece, hard_drop_loc)

    highest_block = 0
    holes_blocked = 0
    column_factor = 0
    for c in range(len(board_list[0])):
        column = board_list[::-1][c]
        count_holes = False
        for r, val in enumerate(column):
            if val != Board_View.Board_Values.EMPTY:
                count_holes = True
                highest_block = max(highest_block, r)
                column_factor += 50
            if count_holes:
                if val == Board_View.Board_Values.EMPTY:
                    holes_blocked += 1
                    column_factor -= 60

    row_holes_multiplier = [2 ** (board_height - i) for i in range(board_height)]

    row_factor = 0
    filled_streak = 1
    blocks_in_prev_row = board_width - 1
    for r, row in enumerate(board_list[::-1]):
        blocks_in_row = 0
        emptys_in_row = 0

        for val in row:
            if val == Board_View.Board_Values.EMPTY:
                emptys_in_row += 1
            else:
                blocks_in_row += 1

        prev_row_multiplier = (blocks_in_prev_row / (board_width - 1))
        current_row_multiplier = (blocks_in_row / (board_width - 1))

        if emptys_in_row == 1:
            filled_streak += 1
        else:
            filled_streak = 1

        row_factor += current_row_multiplier * row_holes_multiplier[r] * prev_row_multiplier * filled_streak

        blocks_in_prev_row = blocks_in_row

    combo_mulitplier = 3

    return score + row_factor \
        + column_factor \
        + combo_mulitplier ** comboCount

def topologicalEvalFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    score = currentGameState.get_score()
    comboCount = currentGameState.get_combo_count()
    queue = currentGameState.get_queue()
    hold = currentGameState.get_hold()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    hard_drop_loc, _ = piece.hard_drop(board, piece_loc)
    board_list = board.asList(piece, hard_drop_loc)

    topology = list()

    for c in range(board_width):
        col = board_list[::-1][c]
        height = 0
        for r, val in enumerate(col):
            if val != Board_View.Board_Values.EMPTY:
                height = board_width - r
                break
        topology.append(height)

    smallest = min(topology)
    avg = (sum(topology) - smallest) / board_width - 1

    topologyQuality = sum(((avg - col) for col in topology if col != smallest))

    return score + 30 * topologyQuality

def get_surrounding(grid):
    return [grid.get_adj(dir) for dir in [Adj_Grid_Names.N,  Adj_Grid_Names.N_E, Adj_Grid_Names.E,
                                          Adj_Grid_Names.S_E, Adj_Grid_Names.S, Adj_Grid_Names.S_W,
                                          Adj_Grid_Names.W, Adj_Grid_Names.N_W]]

def fitSnugEvalFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    score = currentGameState.get_score()
    comboCount = currentGameState.get_combo_count()
    queue = currentGameState.get_queue()
    hold = currentGameState.get_hold()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    hard_drop_loc, _ = piece.hard_drop(board, piece_loc)
    hard_drop_appendages = set(piece.get_appendages(hard_drop_loc))
    board_list = board.asList(piece, hard_drop_loc)

    surrounding_blocks = 0
    for appendage in hard_drop_appendages:
        for surrounding in get_surrounding(appendage):
            if not surrounding in hard_drop_appendages and board[surrounding] != Board_View.Board_Values.EMPTY:
                surrounding_blocks += 1
    
    return score + 25 * surrounding_blocks

def testEvalFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    score = currentGameState.get_score()
    comboCount = currentGameState.get_combo_count()
    queue = currentGameState.get_queue()
    hold = currentGameState.get_hold()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    hard_drop_loc, _ = piece.hard_drop(board, piece_loc)
    hard_drop_appendages = set(piece.get_appendages(hard_drop_loc))
    board_list = board.asList(piece, hard_drop_loc)

    surrounding_blocks = 0
    for appendage in hard_drop_appendages:
        for surrounding in get_surrounding(appendage):
            if not surrounding in hard_drop_appendages and board[surrounding] != Board_View.Board_Values.EMPTY:
                surrounding_blocks += 1

    topology = list()

    for c in range(board_width):
        height = 0
        for r in range(board_height-1, -1, -1):
            if board_list[r][c] != Board_View.Board_Values.EMPTY:
                height = r
                break
        topology.append(height)

    smallest = max(min(topology), 1)
    avg = (sum(topology) - smallest) / board_width - 1

    win_lose_points = 0
    if currentGameState.is_game_over():
        win_lose_points -= 10000

    topologyQuality = sum((((smallest - col) ** 2) * smallest for col in topology if abs(col - smallest) > 1))

    return score - 10 * topologyQuality + 15 * surrounding_blocks + win_lose_points
    

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState, agents):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        depth  = gameState.get_queue_size() * gameState.get_board_size()[1]
        return self.expectimax(gameState, self.index, 2, agents)[1]
    
    def expectimax(self, gameState, agentIndex, depth, agents):
        if depth <= 0 or gameState.is_game_over():
            return (self.evaluationFunction(gameState),)

        if agentIndex == self.index:
            return self.max_val(gameState, agentIndex, depth, agents)
        else:
            return self.exp_val(gameState, agentIndex, depth, agents)

    def max_val(self, gameState, agentIndex, depth, agents):
        next_agent = (agentIndex + 1) % len(agents)
        next_depth = depth if next_agent else depth - 1
        best_action = (-float("inf"), None)
        evalFunc = max
        
        for action in gameState.getLegalActions():
            successor = gameState.getSuccessor(action)
            result = (self.expectimax(successor, next_agent, next_depth, agents)[0], action)
            best_action = evalFunc(result, best_action, key = lambda successor: successor[0])

        return best_action

    def exp_val(self, gameState, agentIndex, depth, agents):
        next_agent = (agentIndex + 1) % len(agents)
        next_depth = depth if next_agent else depth - 1
        expected_value = 0

        successor = gameState.getSuccessor(Controls.IDLE)
            
        expected_value = self.expectimax(successor, next_agent, next_depth, agents)[0]

        return (expected_value,)
    
class IdleMoveAgent(MultiAgentSearchAgent):
    def __init__(self):
        super().__init__()
        self.index = 1
        self.frames_since_last = 1
        self.frames_since_last_place = 0
        self.num_blocks_placed = 0

    def getAction(self, state, agents):
        gravity_frames = gravity_to_frames(state.get_gravity())
        block_ttl = 140
        gravity_frames = max(gravity_frames - (self.frames_since_last_place if self.frames_since_last_place > block_ttl else 0), 1)

        current_blocks_placed = state.get_num_blocks_placed()

        if current_blocks_placed != self.num_blocks_placed:
            self.frames_since_last_place = 0
            self.num_blocks_placed = current_blocks_placed

        action = Controls.IDLE 

        if state.check_just_hard_dropped():
            self.frames_since_last = 1
            self.frames_since_last = 1

        self.frames_since_last = ((self.frames_since_last) % gravity_frames) + 1
        self.frames_since_last_place += 1

        if self.frames_since_last_place >= block_ttl and (Controls.HARD_DROP in state.getLegalActions()):
            self.frames_since_last = 1
            action = Controls.HARD_DROP
        elif self.frames_since_last < gravity_frames:
                action = None

        return action

def betterEvaluationFunction(currentGameState: GameState):
    return currentGameState.get_score()

def scoreEvalFunction(currentGameState):
    return currentGameState.get_score()

class QLearningAgent(MultiAgentSearchAgent):
    def __init__(self):
        self.q_table = {}

    def getAction(self, gameState: GameState, agents):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        #TODO: take the time before and after a successor is taken to evalutate for time.
        legalActions = gameState.getLegalActions()
        return legalActions[np.argmax(self._get_table_value(gameState))]
    
    def train(self, gameState):
        print("Training has Begun!")
        alpha = 0.2
        gamma = 0.8
        epsilon = 0.1

        avg_level, avg_score = 0, 0
        total_time_per_decision = 0
        total_decisions = 0

        runs = float(10)

        for i in range(int(runs)):
            print("--------------------------------------\nStarting run:", (i + 1))
            state = gameState.deepCopy()
            done = False

            gameStartTime = time()
            while not done:
                startTime = time()
                legalActions = state.getLegalActions()
                if uniform(0, 1) < epsilon:
                    action = randint(0, len(legalActions) - 1) # Explore action space
                else:
                    action = np.argmax(self._get_table_value(state)) # Exploit learned values
                
                decision_time = time() - startTime
                total_time_per_decision += decision_time
                total_decisions += 1
                next_state = gameState.getSuccessor(legalActions[action]) 
                reward = next_state.get_score() - state.get_score()
                done = next_state.is_game_over()
                
                old_value = self._get_table_value(state)[action]
                next_max = np.max(self._get_table_value(next_state))
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self._get_table_value(state)[action] = new_value

                state = next_state.getSuccessor(Controls.IDLE)

                if startTime - gameStartTime >= 60 * 10:
                    break

            avg_level += float(state.get_level()) / runs
            avg_score += float(state.get_score()) / runs

            print("Finished run:", (i + 1))
            print("Run", (i+1), "level:", state.get_level())
            print("Run", (i+1), "score:", state.get_score())

        print("Training has Finished!")
        print("Average Level:", avg_level)
        print("Average Score:", avg_score)
        print("Average Decision Time:", (total_time_per_decision / float(total_decisions)))

    def _get_table_value(self, state):
        if not state in self.q_table:
            self.q_table[state] = np.zeros(8)
        return self.q_table[state]