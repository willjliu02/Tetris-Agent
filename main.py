from game_states import GameState
from tetris_gui import Tetris_GUI
from tetris_game import Tetris
from game_agents import *
from keyboard_agent import KeyboardAgent
import pickle

q_learning_file = "q_learning_agent 3.bin"
evalFunc = "moreEfficientHolisticEvaluationFunction"
speedHeuristic = "holesBlockedEvalFunction"
use_q_learning = False
train = True

# aiGame = Tetris(GameState(), Tetris_GUI(), PieceAgent())
player = None
if use_q_learning:
    if not train:
        with open(q_learning_file, "rb") as f:
            player = pickle.load(f)
    else:
        player = QLearningAgent(evalFunc)

    if train:
        #Training
        training_game = GameState()
        player.train(training_game)

        with open(q_learning_file, "wb") as f:
            pickle.dump(player, f)
else:
    player = ExpectimaxAgent(evalFunc)
    # player = SpeedExpectimaxAgent(evalFunc)
    # player = KeyboardAgent()

agents = [player, IdleMoveAgent()]
game = Tetris(GameState(), Tetris_GUI(), agents)
game.run()
    
