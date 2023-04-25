from game_states import *
from tetris_game import Controls, Agent, gravity_to_time
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

def holisticEvaluationFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    # Height Fell
    copy = piece.copy()
    copy.orientation = Block_Orientation.UP
    start_location = board.get_piece_start_loc(copy)

    height_dropped = piece_loc.get_points()[1] - start_location.get_points()[1]

    # Block and Line Clear
    placed_board = board.asList(piece, piece_loc)
    appendages = set(map(lambda grid: grid.get_points(), piece.get_appendages(piece_loc)))

    lines_cleared = 0
    included_in_clear = 0
    adjEmpty = 0 
    rows_with_one_plus_holes = 0

    for r, row in enumerate(placed_board):
        blocks_in_row = 0
        blocks_from_piece = 0
        for c, val, in enumerate(row):
            if (c, r) in appendages:
                blocks_from_piece += 1
            if val != Board_View.Board_Values.EMPTY:
                blocks_in_row += 1

            # Adjacent Empty Blocks
            if c == 0 and row[c] == Board_View.Board_Values.EMPTY:
                adjEmpty += 1
            elif row[c-1] != row[c]:
                adjEmpty += 1
            if c == board_width - 1 and row[c] == Board_View.Board_Values.EMPTY:
                adjEmpty += 1

        if blocks_in_row == board_width:
            lines_cleared += 1
            included_in_clear += blocks_from_piece
        elif blocks_in_row > 0:
            rows_with_one_plus_holes += 1

    block_line_clear =  lines_cleared * included_in_clear

    atopEmpty = 0
    holes_blocked = 0
    valleys = 0
    num_blocks_covering = 0

    for c in range(board_width):
        valley_depth = 0
        is_hole_blocked = False
        for r in range(board_height):
            # Atop Empty Blocks
            if r == 0 and placed_board[r][c] == Board_View.Board_Values.EMPTY:
                atopEmpty += 1
            elif placed_board[r - 1][c] != placed_board[r][c]:
                atopEmpty += 1
            if r == board_height - 1 and placed_board[r][c] == Board_View.Board_Values.EMPTY:
                atopEmpty += 1

            # Holes Blocked
            if r < board_height - 1 and placed_board[r][c] == Board_View.Board_Values.EMPTY and placed_board[r + 1][c] != Board_View.Board_Values.EMPTY:
                holes_blocked += 1

            # Valley Depth
            if r < board_height - 1 and placed_board[r][c] == Board_View.Board_Values.EMPTY:
                is_valley = False
                if c == 0 and placed_board[r][c + 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True
                elif c == board_width - 1 and placed_board[r][c - 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True
                elif placed_board[r][c - 1] != Board_View.Board_Values.EMPTY and placed_board[r][c + 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True

                if is_valley:
                    valley_depth += 1
                    valleys += valley_depth

            # Number of Blocks covering Holes
            if is_hole_blocked and placed_board[r][c] != Board_View.Board_Values.EMPTY:
                num_blocks_covering += 1
            elif placed_board[r][c] == Board_View.Board_Values.EMPTY:
                is_hole_blocked = True

    factors = np.array([height_dropped, adjEmpty, rows_with_one_plus_holes, block_line_clear, 
                        atopEmpty, holes_blocked, valleys, num_blocks_covering])
    factorMultiplier  = np.array([1, 1, 1, 1, 
                                  1, 1, 1, 1])

    stateQuality = - np.dot(factors, factorMultiplier)

    return stateQuality

def moreEfficientHolisticEvaluationFunction(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    # Height Fell
    copy = piece.copy()
    copy.orientation = Block_Orientation.UP
    start_location = board.get_piece_start_loc(copy)

    height_dropped = piece_loc.get_points()[1] - start_location.get_points()[1]

    # Block and Line Clear
    placed_board = board.asList(piece, piece_loc)
    np_board0 = board.get_np_able(piece, piece_loc)
    np_board1 = [row[1:] + [1] for row in np_board0]
    np_board2 = [[1] + row[:-1] for row in np_board0]
    np_board3 = np_board0[1:] + [[0] * board_width]
    np_board4 = [[0] * board_width] + np_board0[:-1]

    np_board = np.array([np_board0, np_board1, np_board2, np_board3, np_board4])
    appendages = map(lambda grid: grid.get_points(), piece.get_appendages(piece_loc))

    included_in_clear = 0
    rows_sum = np.sum(np_board[0], axis = 1)
    full_lines = [rows_sum == board_width]
    num_lines_cleared = int(np.sum(full_lines))
    adjEmpty = int(board_height - np.sum(np_board[0], axis=0)[0] + np.sum([np_board[0] != np_board[1]]))
    rows_with_one_plus_holes = int(np.sum([(rows_sum < board_width) & (rows_sum > 0)]))

    cleared_rows = set(np.nonzero(full_lines)[1])

    for block in appendages:
        if block[1] in cleared_rows:
            included_in_clear += 1

    block_line_clear =  num_lines_cleared * included_in_clear

    atopEmpty = int(board_width - np.sum(np_board[0], axis=1)[0]) + int(np.sum([np_board[0] != np_board[3]]))
    holes_blocked = int(np.sum([(np_board[0] == 0) & (np_board[3] == 1)]))
    valleys = 0
    num_blocks_covering = int(np.sum(np.sum(np_board[0], axis = 0) - np.argmin(np_board[0], axis = 0)))

    for c in range(board_width):
        valley_depth = 0
        for r in range(board_height - 1, -1, -1):
            # Valley Depth
            if r < board_height - 1 and placed_board[r][c] == Board_View.Board_Values.EMPTY:
                is_valley = False
                if c == 0 and placed_board[r][c + 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True
                elif c == board_width - 1 and placed_board[r][c - 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True
                elif placed_board[r][c - 1] != Board_View.Board_Values.EMPTY and placed_board[r][c + 1] != Board_View.Board_Values.EMPTY:
                    is_valley = True

                if is_valley:
                    valley_depth += 1
                    valleys += valley_depth
            else:
                break

    factors = np.array([height_dropped, adjEmpty, rows_with_one_plus_holes, block_line_clear, 
                        atopEmpty, holes_blocked, valleys, num_blocks_covering])
    factorMultiplier  = np.array([1, 1, 1, 1, 
                                  1, 1, 1, 1])

    stateQuality = - np.dot(factors, factorMultiplier)

    return stateQuality

def worthCalculatingHeuristic(currentGameState: GameState):
    piece = currentGameState.get_piece()
    piece_loc = currentGameState.get_piece_loc()
    board = currentGameState.get_board()
    board_height = board.get_height()
    board_width = board.get_width()

    hard_drop_loc, _ = piece.hard_drop(board, piece_loc)

    placed_board = board.asList(piece, hard_drop_loc)

    blocks_placed = 0
    num_blocks_covering = 0

    for c in range(board_width):
        is_hole_blocked = False
        for r in range(board_height):
            if placed_board[r][c] != Board_View.Board_Values.EMPTY:
                blocks_placed += 1

            if is_hole_blocked and placed_board[r][c] != Board_View.Board_Values.EMPTY:
                num_blocks_covering += 1
            elif placed_board[r][c] == Board_View.Board_Values.EMPTY:
                is_hole_blocked = True

    return num_blocks_covering / blocks_placed
    

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState, agents):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        depth = gameState.get_queue_size() * gameState.get_board_size()[1]
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
        idle_prob = 0.20
        drop_prob = 0.01
        
        if gameState.check_can_place():
            successor = gameState.getSuccessor([(1, Controls.PLACE)])
            expected_value += self.expectimax(successor, next_agent, next_depth, agents)[0]
        else:
            for move, prob in [(None, 1 - (idle_prob + drop_prob)), (Controls.IDLE, idle_prob), (Controls.HARD_DROP, drop_prob)]:
                successor = gameState.getSuccessor((1, move))
                expected_value += prob * self.expectimax(successor, next_agent, next_depth, agents)[0]

        return (expected_value,)
    
class SpeedExpectimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='scoreEvaluationFunction', speedHeuristic = "worthCalculatingHeuristic"):
        super().__init__(evalFn)    
        self.heuristic = lookup(speedHeuristic, globals())

    def getAction(self, gameState: GameState, agents):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        depth = gameState.get_queue_size() * gameState.get_board_size()[1]
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

            if self.heuristic(successor) < 0.8:
                result = (self.expectimax(successor, next_agent, next_depth, agents)[0], action)
                best_action = evalFunc(result, best_action, key = lambda successor: successor[0])

        return best_action

    def exp_val(self, gameState, agentIndex, depth, agents):
        next_agent = (agentIndex + 1) % len(agents)
        next_depth = depth if next_agent else depth - 1

        expected_value = 0
        idle_prob = 0.20
        drop_prob = 0.01
        
        if gameState.check_can_place():
            successor = gameState.getSuccessor([(1, Controls.PLACE)])
            expected_value += prob * self.expectimax(successor, next_agent, next_depth, agents)[0]
        else:
            for move, prob in [(None, 1 - (idle_prob + drop_prob)), (Controls.IDLE, idle_prob), (Controls.HARD_DROP, drop_prob)]:
                successor = gameState.getSuccessor((1, move))
                expected_value += prob * self.expectimax(successor, next_agent, next_depth, agents)[0]

        return (expected_value,)
    
class IdleMoveAgent(MultiAgentSearchAgent):
    def __init__(self):
        super().__init__()
        self.index = 1
        self.time_since_last = time()
        self.frames_since_last_place = 0
        self.num_blocks_placed = 0

    def getAction(self, state, agents):
        current_time = time()
        if state.check_can_place():
            self.time_since_last = current_time
            self.frames_since_last_place = 0
            return [(1, Controls.PLACE)]

        gravity_time = gravity_to_time(state.get_gravity())
        block_ttl = 100

        current_blocks_placed = state.get_num_blocks_placed()

        if current_blocks_placed != self.num_blocks_placed:
            self.frames_since_last_place = 0
            self.num_blocks_placed = current_blocks_placed

        action = Controls.IDLE

        self.frames_since_last_place += 1

        if self.frames_since_last_place >= block_ttl and (Controls.HARD_DROP in state.getLegalActions()):
            self.frames_since_last_place = 1
            self.time_since_last = current_time
            action = Controls.HARD_DROP
        elif current_time - self.time_since_last < gravity_time:
                action = None

        return [(1, action)]

def betterEvaluationFunction(currentGameState: GameState):
    return currentGameState.get_score()

def scoreEvalFunction(currentGameState):
    return currentGameState.get_score()

class QLearningAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn = 'scoreEvaluationFunction'):
        super().__init__(evalFn)
        self.q_table = {}

    def getAction(self, gameState: GameState, agents):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        #TODO: take the time before and after a successor is taken to evalutate for time.
        legalActions = gameState.getLegalActions()

        if gameState in self.q_table:
            action = legalActions[np.argmax(self._get_table_value(gameState))]
        else:
            action = legalActions[randint(0, len(legalActions) - 1)]
        return action
    
    def train(self, gameState, track = False):
        print("Training has Begun!")
        alpha = 0.1
        gamma = 0.8
        epsilon = 0.4

        total_level, total_score = 0, 0
        total_time_per_decision = 0
        total_decisions = 0
        total_runs = 0

        runs = 10
        minutes_per_run = 3
        refresh_rate = max(int(runs * 0.1), 20)

        display_rate = refresh_rate

        if track:
            display_rate = 1

        for i in range(int(runs)):
            if i % display_rate == 0:
                print("-" * 30)
                print("Starting run:", (i + 1))

            if i % refresh_rate == 0:
                gameState.setup()

            state = gameState.deepCopy()
            done = False
            start_time_per_decision = total_time_per_decision
            start_decision_count = total_decisions
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
                action_to_take = legalActions[action]
                next_state = state.getSuccessor(action_to_take) 

                if uniform(0, 1) < 0.4:
                    next_state = next_state.getSuccessor(Controls.IDLE)

                reward = self.evaluationFunction(next_state)
                done = next_state.is_game_over()
                
                old_value = self._get_table_value(state)[action]
                next_max = np.max(self._get_table_value(next_state))
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self._get_table_value(state)[action] = new_value

                state = next_state

                if startTime - gameStartTime >= 60 * minutes_per_run:
                    break
            
            total_runs += 1
            total_level += state.get_level()
            total_score += state.get_score()

            if i % display_rate == 0:
                print("Finished run:", (i + 1))
                if track:
                    print("\nRun", (i+1), "level:", state.get_level())
                    print("Run", (i+1), "score:", state.get_score())
                    print("Run", (i+1), "average move time:", ((total_time_per_decision - start_time_per_decision) / (total_decisions - start_decision_count)))
                else:
                    print("Current Average Level:", (total_level / total_runs))
                    print("Current Average Score:", (total_score / total_runs))

        print("-" * 30)
        print("Training has Finished!")
        print("Average Level:", (total_level / total_runs))
        print("Average Score:", (total_score / total_runs))
        print("Average Decision Time:", (total_time_per_decision / float(total_decisions)))

    def _get_table_value(self, state):
        if not state in self.q_table:
            self.q_table[state] = np.zeros(len(state.getLegalActions()))
        return self.q_table[state]