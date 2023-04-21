from graphicsUtils import *
import math, time
from util import Grid
from tetris_model import Board_View
from blocks import *

WINDOW_HEIGHT = 720
WINDOW_WIDTH = 960
GRID_SIZE = 32
BLOCK_SIZE = GRID_SIZE/2
COLOR_WHITE = formatColor(1, 1, 1)
COLOR_BLACK = formatColor(0, 0, 0)
BACKGROUND_COLOR = COLOR_WHITE
WALL_COLOR = formatColor(0.0/255.0, 51.0/255.0, 255.0/255.0)
BLOCK_EDGE_COLORS = formatColor(150/255,150/255,150/255)
SCROLL_HOLD_PANE_COLOR = formatColor(.9, .9, .9)
QUEUE_PANE_COLOR = formatColor(.9, .9, .9)
SCORE_COLOR = formatColor(150/255,150/255,150/255)
BLOCK_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

BOARD_VAL_TO_COLOR = {Board_View.Board_Values.T: formatColor(102/255,0,204/255),
                      Board_View.Board_Values.L: formatColor(1,153/255,0),
                      Board_View.Board_Values.REV_L: formatColor(51/255,51/255,1),
                      Board_View.Board_Values.Z: formatColor(1,26/255,26/255),
                      Board_View.Board_Values.S: formatColor(0,204/255,0),
                      Board_View.Board_Values.LONG: formatColor(51/255,204/255,1),
                      Board_View.Board_Values.SQUARE: formatColor(1,1,0),
                      Board_View.Board_Values.EMPTY: COLOR_BLACK}

BLOCK_SIZE = 0.48
SCARED_COLOR = formatColor(1,1,1)

BLOCK_VEC_COLORS = [colorToVector(BOARD_VAL_TO_COLOR[c]) for c in BOARD_VAL_TO_COLOR]

def piece_to_color(block):
    return_val = None
    if isinstance(block, T_Block):
        return_val = Board_View.Board_Values.T
    elif isinstance(block, L_Block):
        return_val = Board_View.Board_Values.L
    elif isinstance(block, Rev_L_Block):
        return_val = Board_View.Board_Values.REV_L
    elif isinstance(block, Z_Block):
        return_val = Board_View.Board_Values.Z
    elif isinstance(block, S_Block):
        return_val = Board_View.Board_Values.S
    elif isinstance(block, Long_Block):
        return_val = Board_View.Board_Values.LONG
    elif isinstance(block, Square_Block):
        return_val = Board_View.Board_Values.SQUARE
    return BOARD_VAL_TO_COLOR[return_val]

def adjustAppendages(blocks):
    shift_vector = (blocks[0] - blocks[1])/2

    shift_blocks = list(map(lambda appendage:
                            (appendage.shift(shift_vector.get_points())).get_points(),
                          blocks))
    
    mirrored_blocks = list()
    mirror_r = shift_blocks[0][1]
    for block in shift_blocks:
        new_r = mirror_r - block[1]
        mirrored_blocks.append((block[0], mirror_r + new_r))

    return mirrored_blocks

class ScoreHoldPane:
    def __init__(self):
        self.width = WINDOW_WIDTH/8
        self.height = WINDOW_HEIGHT
        self.gridSize = GRID_SIZE
        self.blockSize = self.gridSize // 2
        self.fontSize = 24
        self.textColor = SCORE_COLOR
        self.drawPane()

    def toScreen(self, pos, y = None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x,y = pos
        else:
            x = pos

        x = self.gridSize * (x + 1) # +1 for Padding
        y = self.gridSize * (y + 1)
        return x,y

    def drawPane(self):
        self.scoreText = text( self.toScreen(1, 18  ), self.textColor, "SCORE:      0", "Times", self.fontSize, "bold")
        self.levelText = text( self.toScreen(1, 17  ), self.textColor, "LEVEL:      1", "Times", self.fontSize, "bold")
        self.holdText = text( self.toScreen(1.4, 1  ), self.textColor, "HOLD", "Times", self.fontSize, "bold")
        holdX = 2.5
        holdY = 3
        self.holdCenter = (holdX, holdY)  
        holdCenterToScreen = self.toScreen(holdX, holdY)
        holdRs = 2.5*self.gridSize
        self.holdBox = square(holdCenterToScreen, holdRs, 
                              COLOR_BLACK, COLOR_WHITE)
        self.holdBlocks = None
        # TODO: Display combos (when gotten)

    def updateScore(self, score):
        changeText(self.scoreText, "SCORE: % 6d" % score)

    def updatelevel(self, level):
        changeText(self.levelText, "LEVEL: % 6d" % level)

    def updateHold(self, block):
        appendages = adjustAppendages(block.get_appendages(Grid(self.holdCenter)))

        if self.holdBlocks is None:
            self.holdBlocks = list(map(lambda holdBlock:
                                            square(self.toScreen(holdBlock), self.blockSize, piece_to_color(block), COLOR_WHITE),
                                        appendages))
        else:
            block_color = piece_to_color(block)
            for blockID, new_pos in zip(self.holdBlocks, appendages):
                moveSquare(blockID, self.toScreen(new_pos), self.blockSize)
                changeColor(blockID, block_color)
        
    def displayCombo(self, comboStreak):
        # TODO: display the number of combos for like X number of clicks
        pass

    def drawWarning(self):
        pass

    def clearIcon(self):
        pass

    def updateMessage(self, message):
        pass

    def clearMessage(self):
        pass

class QueuePane:
    def __init__(self, queue_size = 3):
        self.width = WINDOW_WIDTH/4
        self.height = WINDOW_HEIGHT
        self.gridSize = GRID_SIZE
        self.blockSize = self.gridSize // 2
        self.xBase = WINDOW_WIDTH - self.width
        self.fontSize = 24
        self.textColor = SCORE_COLOR
        self.queue_size = queue_size
        self.drawPane()

    def toScreen(self, pos, y = None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x,y = pos
        else:
            x = pos

        x = self.gridSize * (x + 1) + self.xBase # +1 for Padding
        y =  self.gridSize * (y + 1)
        return x,y

    def drawPane(self):
        self.nextText = text( self.toScreen(1, 1  ), self.textColor, "NEXT", "Times", self.fontSize, "bold")
        queueX = 2
        queueY = 3
        self.queueCenters = [(queueX, queueY + (4 * i)) for i in range (self.queue_size)]
        queueCentersToScreen = list(map(lambda grid:
                                            self.toScreen(grid),
                                        self.queueCenters))
        holdRs = 2.5*self.gridSize
        self.queueBoxes = list(map(lambda boxCenter:
                                        square(boxCenter, holdRs, COLOR_BLACK, COLOR_BLACK),
                                    queueCentersToScreen))
        self.queueBlocks = None

    def updateQueue(self, queue):
        appendages = list(map(lambda piece, boxCenter:
                            adjustAppendages(piece.get_appendages(Grid(boxCenter))),
                            queue, self.queueCenters))

        if self.queueBlocks is None:
            self.queueBlocks = list(map(lambda blockCenters, block:
                                            list(map(lambda blockCenter:
                                                        square(self.toScreen(blockCenter), self.blockSize, piece_to_color(block), COLOR_WHITE),
                                                    blockCenters)),
                                    appendages, queue))
        else:
            for i, (queueBlockIDs, blockLocs) in enumerate(zip(self.queueBlocks, appendages)):
                block_color = piece_to_color(queue[i])
                for blockID, new_pos in zip(queueBlockIDs, blockLocs):
                    moveSquare(blockID, self.toScreen(new_pos), self.blockSize)
                    changeColor(blockID, block_color)
        
class Board_Pane:
    def __init__(self, board_width = 10, board_height = 20) -> None:
        self.width = WINDOW_WIDTH/2
        self.height = WINDOW_HEIGHT
        self.gridSize = GRID_SIZE
        self.blockSize = self.gridSize // 2
        self.xBase = WINDOW_WIDTH/5
        self.fontSize = 24
        self.textColor = SCORE_COLOR
        self.board_height = board_height
        self.board_width = board_width
        self.drawPane()

    def toScreen(self, pos, y = None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x,y = pos
        else:
            x = pos

        x = self.gridSize * (x + 1) + self.xBase # +1 for Padding
        y =  self.gridSize * (y + 1)
        return x,y

    def drawPane(self):
        leftX = 3
        bottomY = 20
        self.boardBlocksLocs = [[(leftX + dx, bottomY - dy) for dx in range (self.board_width)] for dy in range(self.board_height)]
        boardBlocksToScreen = list(map(lambda row:
                                            list(map(lambda block:
                                                        self.toScreen(block),
                                                    row)),
                                        self.boardBlocksLocs))
        self.boardBlocks = list(map(lambda boardBlockRows:
                                    list(map(lambda boxCenter:
                                        square(boxCenter, self.blockSize, COLOR_BLACK, BLOCK_EDGE_COLORS),
                                            boardBlockRows)),
                                    boardBlocksToScreen))

    def updateMovingPiece(self, board, from_blocks, to_blocks):
        to_blocks = set(to_blocks)
        change_black = [block for block in from_blocks if not block in to_blocks]

        for block in change_black:
            c, r = block.get_points()
            changeColor(self.boardBlocks[r][c], COLOR_BLACK)

        for block in to_blocks:
            c, r = block.get_points()
            changeColor(self.boardBlocks[r][c], BOARD_VAL_TO_COLOR[board[r][c]])
    
    def updateClearFrom(self, board, clear_row):
        for r in range(clear_row, board.get_height()):
            for c in range(board.get_width()):
                changeColor(self.board[r][c], BOARD_VAL_TO_COLOR[board[r][c]])

    def updateBoard(self, board):
        for r, row in enumerate(self.boardBlocks):
            for c, block in enumerate(row):
                changeColor(block, BOARD_VAL_TO_COLOR[board[r][c]])

class Tetris_GUI:
    def __init__(self) -> None:
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT

    def initialize(self, board_width = 10, board_height = 20, queue_size = 3):
        self.leftPanel = ScoreHoldPane()
        self.boardPanel = Board_Pane(board_width, board_height)
        self.rightPanel = QueuePane(queue_size)

    def make_window(self, window_title = "CS 4100"):
        begin_graphics(self.window_width, self.window_height, BACKGROUND_COLOR, window_title)

    def update_current_piece(self, board, from_blocks, to_blocks):
        self.boardPanel.updateMovingPiece(board, from_blocks, to_blocks)

    def update(self, state):
        state_data = state.data
        board, hold, queue, score, level = state.get_gui_board(), state_data.hold, state_data.queue, state_data.score, state_data.level

        if not hold is None:
            self.leftPanel.updateHold(hold)

        self.leftPanel.updateScore(score)

        self.rightPanel.updateQueue(queue)

        self.boardPanel.updateBoard(board)

        self.leftPanel.updatelevel(level)

    def movePiece(self, board, from_blocks, to_blocks):
        self.boardPanel.updateMovingPiece(board, from_blocks, to_blocks)

    def refreshClears(self, board, bottom_cleared_row):
        self.boardPanel.updateClearFrom(board, bottom_cleared_row)

    def close_graphics(self):
        end_graphics()