from game_states import GameState
from tetris_gui import Tetris_GUI
from tetris_game import Tetris
from game_agents import *
from keyboard_agent import KeyboardAgent
import pickle

q_learning_file = "q_learning_agent.bin"
use_q_learning = True

# aiGame = Tetris(GameState(), Tetris_GUI(), PieceAgent())
player = None
if use_q_learning:
    player = QLearningAgent()

    #Training
    training_game = GameState()
    training_game.setup()
    player.train(training_game)

    with open(q_learning_file, "wb") as f:
        pickle.dump(player, f)

else:
    player = ExpectimaxAgent("scoreEvaluationFunction")
# player = KeyboardAgent()
agents = [player, IdleMoveAgent()]
game = Tetris(GameState(), Tetris_GUI(), agents)
game.run()
    
