from util import Grid, Adj_Grid_Names
from blocks import *
from enum import IntEnum
from tetris_game import Controls

def get_level_line_clears(level):
    if level < 10:
        return (level + 1) * 10
    elif level < 16:
        return 100
    elif level < 25:
        return (level - 5) * 10
    else:
        return 200
    
def level_to_gravity(level):
    return level

class Board_View:
    class Board_Values(IntEnum):
        BOUND = 10
        EMPTY = 0
        T = 1
        L = 2
        REV_L = 3
        Z = 4
        S = 5
        SQUARE = 6
        LONG = 7
        def getBoardValue(block):
            if isinstance(block, T_Block):
                return Board_View.Board_Values.T
            elif isinstance(block, L_Block):
                return Board_View.Board_Values.L
            elif isinstance(block, Rev_L_Block):
                return Board_View.Board_Values.REV_L
            elif isinstance(block, S_Block):
                return Board_View.Board_Values.S
            elif isinstance(block, Z_Block):
                return Board_View.Board_Values.Z
            elif isinstance(block, Square_Block):
                return Board_View.Board_Values.SQUARE
            elif isinstance(block, Long_Block):
                return Board_View.Board_Values.LONG
            else:
                raise ValueError
            

    def __init__(self, board) -> None:
        self.board = board

    def __getitem__(self, key:Grid):
        c, r = key.c, key.r

        if r >= self.board.get_height() - 2:
            return Board.Board_Values.BOUND

        return self.board[key]
    
    def get_height(self):
        return self.board.get_height() - 2
    
    def get_width(self):
        return self.board.get_width()

class Board(Board_View):
    '''
    Grid orientation is bottom-left is 0, 0 and top-right is width - 1, height - 1
    '''
    def __init__(self, width, height, board = None) -> None:
        self.height = height # an unseen buffer on the top of the board
        self.width = width
        self.board = self.makeBoard() if board is None else [row.copy() for row in board]

    def get_piece_start_loc(self, piece):
        if piece is None:
            raise ValueError
        
        new_piece_r = self.height - 4
        new_piece_c = self.width // 2
        piece_grid = Grid(new_piece_c, new_piece_r)

        while not piece.can_move(self, piece_grid) and piece_grid.r < self.height-2:
            piece_grid = piece_grid.get_adj(Adj_Grid_Names.N)
        
        return piece_grid

    def _can_be_placed(self, piece, loc):
        piece_blocks = piece.get_appendages(loc)

        for block in piece_blocks:
            if self[block] != Board_View.Board_Values.EMPTY:
                return False

        return True

    '''
    Piece has been checked and it valid at this position
    '''
    def place_piece(self, piece, loc):
        piece_blocks = piece.get_appendages(loc)

        for block in piece_blocks:
            self[block] = Board_View.Board_Values.getBoardValue(piece)

        return True

    def clear_lines(self):
        lines_cleared = 0
        new_board = list()

        for row in self.board:
            if row.is_clearable():
                lines_cleared += 1
            else:
                new_board.append(row)

        for i in range(lines_cleared):
            new_board.append(Board.Row(self.width))

        self.board = new_board

        return lines_cleared

    class Row:
        def __init__(self, width, counter = 0, row = None) -> None:
            self.width = width
            self.counter = counter
            self.row = [Board.Board_Values.EMPTY for i in range(self.width)] if row is None else [ele for ele in row]

        def _add_block(self, index, piece_val):
            self.row[index] = piece_val
            self.counter += 1

        def __setitem__(self, key, newvalue):
            self._add_block(key, newvalue)

        def __getitem__(self, key):
            return self.row[key]
        
        def is_clearable(self):
            return self.counter >= self.width
        
        def get_width(self):
            return self.width
        
        def get_count(self):
            return self.counter
        
        def copy(self):
            return Board.Row(self.width, self.counter, self.row)

    def makeBoard(self):
        return [Board.Row(self.width) for i in range(self.height)]

    def get_height(self):
        return self.height - 2
    
    def get_width(self):
        return self.width

    def __setitem__(self, key:Grid, new_value):
        c, r = key.c, key.r

        if c < 0 or r < 0 or r >= self.get_height() or c >= self.get_width():
            return 
        
        self.board[r][c] = new_value

    def __getitem__(self, key:Grid):
        c, r = key.c, key.r

        if c < 0 or r < 0 or r >= self.get_height() or c >= self.get_width():
            return Board.Board_Values.BOUND
        
        return self.board[r][c]
    
    def deepCopy(self):
        return Board(self.width, self.height, self.board)
    
    def asList(self, piece, loc):
        piece_grids = set(map(lambda grid: grid.get_points(), piece.get_appendages(loc)))
        board = list()
        for r in range(self.get_height()):
            board_row = list()
            for c in range(self.get_width()):
                grid = Grid(c, r)
                if (c, r) in piece_grids:
                    board_row.append(Board_View.Board_Values.getBoardValue(piece))
                else:
                    board_row.append(self[grid])
            board.append(board_row)

        return board
    
    def get_row(self, row):
        return self.board[row].copy()