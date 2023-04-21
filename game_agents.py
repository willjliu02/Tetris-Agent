from game_states import *
from tetris_game import Controls, Agent, gravity_to_frames
from util import Adj_Grid_Names, lookup
from blocks import turn
from time import time

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
    height_multiplier = 20

    return score + row_factor \
        + column_factor \
        + combo_mulitplier ** comboCount

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

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
        gravity_frames = max(gravity_frames - self.frames_since_last_place if self.frames_since_last_place > 120 else 0, 1)

        current_blocks_placed = state.get_num_blocks_placed()

        if current_blocks_placed != self.num_blocks_placed:
            self.frames_since_last_place = 0
            self.num_blocks_placed = current_blocks_placed

        action = Controls.IDLE 

        if state.check_just_hard_dropped():
            self.frames_since_last = 1

        self.frames_since_last = ((self.frames_since_last) % gravity_frames) + 1
        self.frames_since_last_place += 1

        if self.frames_since_last_place >= 120 and (Controls.HARD_DROP in state.getLegalActions()):
            self.frames_since_last = 1
            action = Controls.HARD_DROP
        if self.frames_since_last < gravity_frames:
                action = None

        return action

def betterEvaluationFunction(currentGameState: GameState):
    return currentGameState.get_score()

def scoreEvalFunction(currentGameState):
    return currentGameState.get_score()