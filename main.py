from game_states import GameState
from tetris_gui import Tetris_GUI
from tetris_game import Tetris
from game_agents import *
from keyboard_agent import KeyboardAgent
import pickle
import argparse
import sys

parser = argparse.ArgumentParser(description='Run the game Tetris')
parser.add_argument('-a', '--agent', choices=["q-learn", 'expectimax', 'keyboard'], default = "keyboard", dest="agent",
                    nargs="?", type = str)
parser.add_argument('-t', '--train', action="store_true")
parser.add_argument('-r', '--runs', nargs = "?", default=200, type=int)
parser.add_argument('-d', '--depth', nargs = "?", default=2, type=str)
parser.add_argument('-l', '--load', default=None, type=str, nargs="?")
parser.add_argument('-s', '--save', default="out.bin", type=str, nargs="?")
parser.add_argument('-e', '--eval', default="moreEfficientHolisticEvaluationFunction", type=str, nargs="?")

args = parser.parse_args()

evalFunc = args.eval

if args.agent == "q-learn":
    use_q_learning = True
    train = args.train

    if train:
        save_file = args.save
    else:
        save_file = None

    if args.load is None:
        use_existing = False
    else:
        use_existing = True
        load_file = args.load
else:
    use_q_learning = False

# aiGame = Tetris(GameState(), Tetris_GUI(), PieceAgent())
player = None
if use_q_learning:
    if use_existing:
        print("Loading Trained Player")
        with open(load_file, "rb") as f:
            player = pickle.load(f)
        print("Finished Loading Trained Player")
    else:
        player = QLearningAgent(evalFunc)

    if train:
        #Training
        print("Training Player")
        training_game = GameState()
        player.train(training_game, runs = args.runs, track=False)
        print("Finished Training Player")

        print("Saving Trained Player")
        with open(save_file, "wb") as f:
            pickle.dump(player, f)
        print("Finished Saving Trained Player")
else:
    if args.agent == "expectimax":
        player = ExpectimaxAgent(evalFunc, args.depth)
        # player = SpeedExpectimaxAgent(evalFunc)
    else:
        player = KeyboardAgent()

agents = [player, IdleMoveAgent()]
game = Tetris(GameState(), Tetris_GUI(), agents)
game.run()
    
