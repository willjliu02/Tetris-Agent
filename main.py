from game_states import GameState
from tetris_gui import Tetris_GUI
from tetris_game import Tetris
from game_agents import IdleMoveAgent, ExpectimaxAgent
from keyboard_agent import KeyboardAgent

# aiGame = Tetris(GameState(), Tetris_GUI(), PieceAgent())
player = ExpectimaxAgent()
# player = KeyboardAgent()
agents = [player, IdleMoveAgent()]
game = Tetris(GameState(), Tetris_GUI(), agents)
game.run()
