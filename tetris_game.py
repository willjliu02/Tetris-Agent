from enum import IntEnum
import time

class Controls(IntEnum):
    IDLE = 0
    ROTATE_LEFT = 1
    ROTATE_RIGHT = 2
    SOFT_DROP = 3
    HARD_DROP = 4
    MOVE_LEFT = 5
    MOVE_RIGHT = 6
    HOLD = 7
    PLACE = 8

def gravity_to_time(gravity):
    return 0.9
    
class GameStateData:
    def __init__(self, prevState = None):
        if not prevState is None:
            self.hold = prevState.hold.copy() if prevState.hold else None
            self.just_held = prevState.just_held
            self.comboCount = prevState.comboCount
            self.queue = tuple((piece.copy() for piece in prevState.queue))
            self.current_piece = prevState.current_piece.copy()
            self.board = prevState.board.deepCopy()
            self.level = prevState.level
            self.lines_cleared_on_lvl = prevState.lines_cleared_on_lvl
            self.lines_to_clear_on_lvl = prevState.lines_to_clear_on_lvl
            self.gameover = prevState.gameover
            self.score = prevState.score
            self.current_piece_loc = prevState.current_piece_loc.copy()
            self.gravity = prevState.gravity
            self.placed_blocks = prevState.placed_blocks
            self.move_history = tuple(((piece.copy(), loc.copy()) for (piece, loc) in prevState.move_history))
            self.can_place = prevState.can_place
            
        else:
            self.hold = None
            self.just_held = False
            self.comboCount = 0
            self.queue = None
            self.current_piece = None
            self.board = None
            self.score = 0
            self.level = 1
            self.lines_cleared_on_lvl = 0
            self.lines_to_clear_on_lvl = 10
            self.gameover = False
            self.current_piece_loc = None
            self.gravity = 1
            self.placed_blocks = 0
            self.move_history = None
            self.can_place = False
            
    def initialize(self, piece, piece_loc, queue, board = None, move_history = None):
        if not board is None:
            self.board = board

        if not move_history is None:
            self.move_history = move_history  

        self.current_piece = piece
        self.just_held = False
        self.queue = queue
        self.current_piece_loc = piece_loc

    def deepCopy(self):
        return GameStateData(self)
    
    def __hash__(self) -> int:
        hashBoard = tuple((tuple(row) for row in self.board.asList()))
        topology = tuple(self.board.get_topology())
        min_height = min(topology)
        topology = tuple((height - min_height for height in topology))
        hashed_items = (self.current_piece, hashBoard, self.queue[0], self.hold)
        return hash(hashed_items)

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        pass

class Tetris:
    def __init__(self, game, view, agents = None, move_time = 5000, catch_exception = True) -> None:
        self.game = game
        self.view = view
        self.catch_exception = catch_exception
        self.agents = agents
        self.move_time = move_time
        self.move_history = list()

    def run(self):
        self.game.setup()
        self.view.make_window()
        self.view.initialize()

        num_agent = len(self.agents)
        agentIndex = 0

        agents_actions = [[], []]
        agents_action_marker = [0, 0]
        agents_action_execution_times = [0, 0]

        print("Game is Starting!")
        action = None
        while not self.game.is_game_over():
            # Solicit an action
            
            skip_action = False
            observation = self.game.deepCopy()

            if agents_action_execution_times[agentIndex] == 0 and agents_action_marker[agentIndex] == len(agents_actions[agentIndex]):
                agents_actions[agentIndex] = self.agents[agentIndex].getAction(observation, self.agents)
                agents_action_execution_times[agentIndex] = 0
                agents_action_marker[agentIndex] = 0
                action = agents_actions[agentIndex][agents_action_marker[agentIndex]]
                while action[0] == 0:
                    agents_action_marker[agentIndex] += 1
                    action = agents_actions[agentIndex][agents_action_marker[agentIndex]]
                
            action = agents_actions[agentIndex][agents_action_marker[agentIndex]]
            agents_action_execution_times[agentIndex] += 1
            if agents_action_execution_times[agentIndex] >= action[0]:
                agents_action_execution_times[agentIndex] = 0
                agents_action_marker[agentIndex] += 1

            # Execute the action
            if not action[1] is None:
                self.move_history.append( action[1] )

            self.game = self.game.getSuccessor( [(1, action[1])] )

            # Change the display
            self.view.update( self.game )

            agentIndex = (agentIndex + 1) % num_agent

        print("Game Over!")
        print("Final Score:", self.game.get_score())
        print("Final Level:", self.game.get_level())
        self.view.close_graphics()