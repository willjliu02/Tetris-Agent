from abc import ABC
from util import Grid, Adj_Grid_Names, shift_blocks, Vector
from enum import IntEnum
from tetris_game import Controls

class Block_Orientation(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

turn_left = {Block_Orientation.RIGHT: Block_Orientation.UP,
             Block_Orientation.UP: Block_Orientation.LEFT,
             Block_Orientation.LEFT: Block_Orientation.DOWN,
             Block_Orientation.DOWN: Block_Orientation.RIGHT}

turn_right = {Block_Orientation.RIGHT: Block_Orientation.DOWN,
             Block_Orientation.UP: Block_Orientation.RIGHT,
             Block_Orientation.LEFT: Block_Orientation.UP,
             Block_Orientation.DOWN: Block_Orientation.LEFT}

def turn(orientation, rotation):
    if not isinstance(rotation, Controls):
        raise TypeError
    
    if rotation == Controls.ROTATE_LEFT:
        return turn_left[orientation]
    else:
        return turn_right[orientation]

class Block(ABC):
    def __init__(self, existing_block = None) -> None:
        super().__init__()
        if existing_block is None:
            self.orientation = Block_Orientation.UP
        else:
            self.orientation = existing_block.orientation

    def can_move(self, board, to_grid: Grid, orientation = None):
        dest_blocks = self.get_appendages(to_grid, orientation)

        for dest_grid in dest_blocks:
            if board[dest_grid]:
                return False
            
        return True

    '''
    Rotates if possible, then returns the grid its center should be in
    '''
    def rotate(self, board, loc: Grid, rotation):
        new_orientation = turn(self.orientation, rotation)

        appendages = self.get_appendages(loc)
        new_pos = None

        for i, pos in enumerate(appendages):
            new_pos = self._can_rotate(board, pos, new_orientation, i)
            
            if new_pos:
                break

        if new_pos:
            self.orientation = new_orientation

        return new_pos if new_pos else loc
    
    def _can_rotate(self, board, loc, new_orientation, block):
        if block == 0:
            return loc if self.can_move(board, loc, new_orientation) else None

        appendages = self.get_appendages(loc, new_orientation)

        shift_vector = loc - appendages[block]
        appendages = shift_blocks(appendages, shift_vector)

        for appendage in appendages:
            if board[appendage]:
                return None
            
        return appendages[0]
    
    def get_orientation(self):
        return self.orientation

    '''
    Gets the other blocks that break off from the middle
    '''
    def get_appendages(self, to_grid, orientation = None):
        pass

    def copy(self):
        pass

    '''
    Resets the block to be placed into hold
    '''
    def hold(self):
        self.orientation = Block_Orientation.UP
        return self

    '''
    Returns the lowest position its center can be dropped to
    '''
    def hard_drop(self, board, loc:Grid, orientation = None):
        blocks_dropped = 0
        farthest_loc = loc
        test_loc = loc.get_adj(Adj_Grid_Names.S)

        while self.can_move(board, test_loc, orientation):
            blocks_dropped += 1
            farthest_loc = test_loc
            test_loc = test_loc.get_adj(Adj_Grid_Names.S)

        return farthest_loc, blocks_dropped
    
class T_Block(Block):
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.W),grid.get_adj(Adj_Grid_Names.E)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.E), grid.get_adj(Adj_Grid_Names.W)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N)])}
    
    def get_appendages(self, to_grid, orientation = None):
        return T_Block.blocks_from_center[orientation if orientation else self.orientation ](to_grid)
    def copy(self):
        return T_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, T_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 1))
    
class L_Block(Block):
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.W),grid.get_adj(Adj_Grid_Names.N_E)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S_E)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.E), grid.get_adj(Adj_Grid_Names.S_W)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N_W)])}

    def get_appendages(self, to_grid, orientation = None):
        return L_Block.blocks_from_center[orientation if orientation else self.orientation](to_grid)
    def copy(self):
        return L_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, L_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 2))

class Rev_L_Block(Block):
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.E),grid.get_adj(Adj_Grid_Names.N_W)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N_E)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.W), grid.get_adj(Adj_Grid_Names.S_E)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S_W)])}

    def get_appendages(self, to_grid, orientation = None):
        return Rev_L_Block.blocks_from_center[orientation if orientation else self.orientation](to_grid)
    def copy(self):
        return Rev_L_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Rev_L_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 3))

class Square_Block(Block):
    '''
    Center will always be the bottom left "block" of the square
    '''
    def rotate(self, board, loc: Grid, rotation):
        return loc
    
    def get_appendages(self, to_grid, orientation = None):
        return tuple([to_grid.get_adj(Adj_Grid_Names.CENTER), to_grid.get_adj(Adj_Grid_Names.N_E), 
                      to_grid.get_adj(Adj_Grid_Names.E),to_grid.get_adj(Adj_Grid_Names.N)])
    def copy(self):
        return Square_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Square_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 4))

class S_Block(Block):
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.W),grid.get_adj(Adj_Grid_Names.N_E)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S_E)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.E), grid.get_adj(Adj_Grid_Names.S_W)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N_W)])}

    def get_appendages(self, to_grid, orientation = None):
        return S_Block.blocks_from_center[orientation if orientation else self.orientation](to_grid)
    def copy(self):
        return S_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, S_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 5))

class Z_Block(Block):
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.E),grid.get_adj(Adj_Grid_Names.N_W)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N_E)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.W), grid.get_adj(Adj_Grid_Names.S_E)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S_W)])}

    def get_appendages(self, to_grid, orientation = None):
        return Z_Block.blocks_from_center[orientation if orientation else self.orientation](to_grid)
    def copy(self):
        return Z_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Z_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 6))


class Long_Block(Block):
    '''
    Center will be 2nd from the left in the UP Orientation
    '''
    blocks_from_center = {Block_Orientation.UP: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.E), 
                                   grid.get_adj(Adj_Grid_Names.W), grid.get_adj(Adj_Grid_Names.E, Adj_Grid_Names.E)]),
                          Block_Orientation.RIGHT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.S), 
                                   grid.get_adj(Adj_Grid_Names.N), grid.get_adj(Adj_Grid_Names.S, Adj_Grid_Names.S)]),
                          Block_Orientation.DOWN: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.W), 
                                   grid.get_adj(Adj_Grid_Names.E), grid.get_adj(Adj_Grid_Names.W, Adj_Grid_Names.W)]),
                          Block_Orientation.LEFT: lambda grid: 
                            tuple([grid.get_adj(Adj_Grid_Names.CENTER), grid.get_adj(Adj_Grid_Names.N), 
                                   grid.get_adj(Adj_Grid_Names.S), grid.get_adj(Adj_Grid_Names.N, Adj_Grid_Names.N)])}

    def rotate(self, board, loc: Grid, rotation):
        new_orientation = turn(self.orientation, rotation)

        appendages = self.get_appendages(loc)
        new_pos = None

        shift_vector = None

        if rotation == Controls.ROTATE_RIGHT:
            shift_vector = appendages[1] - appendages[0]
        else:
            temp = self.get_appendages(loc, new_orientation)
            shift_vector = temp[0] - temp[1]

        appendages = shift_blocks(appendages, shift_vector)

        for i, pos in enumerate(appendages):
            new_pos = self._can_rotate(board, pos, new_orientation, i)
            
            if new_pos:
                break

        if new_pos:
            self.orientation = new_orientation

        return new_pos if new_pos else loc
    
    def get_appendages(self, to_grid, orientation = None):
        return Long_Block.blocks_from_center[orientation if orientation else self.orientation](to_grid)
    def copy(self):
        return Long_Block(self)
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Long_Block) and self.orientation == __value.orientation
    def __hash__(self) -> int:
        return hash((self.orientation, 7))