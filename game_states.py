from tetris_model import Board, get_level_line_clears, level_to_gravity, Board_View
from tetris_game import Controls, GameStateData
from util import Adj_Grid_Names
from blocks import *
from random import randint, random

GAME_PIECES = [lambda: T_Block(),
                lambda: Z_Block(),
                lambda: S_Block(),
                lambda: L_Block(),
                lambda: Rev_L_Block(),
                lambda: Long_Block(),
                lambda: Square_Block()]



class GameState:
    control_to_move = {Controls.HARD_DROP: lambda game: game.hard_drop(),
                       Controls.SOFT_DROP: lambda game: game._move_dir(Adj_Grid_Names.S),
                       Controls.MOVE_LEFT: lambda game: game._move_dir(Adj_Grid_Names.W),
                       Controls.MOVE_RIGHT: lambda game: game._move_dir(Adj_Grid_Names.E),
                       Controls.ROTATE_LEFT: lambda game: game._rotate(Controls.ROTATE_LEFT),
                       Controls.ROTATE_RIGHT: lambda game: game._rotate(Controls.ROTATE_RIGHT),
                       Controls.IDLE: lambda game: game._move_idley(),
                       Controls.HOLD: lambda game: game.hold()}

    def __init__(self, prevState = None) -> None:
        if not prevState is None:
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def setup(self, board_width = 10, board_height = 20, queue_size = 3):
        current_piece = self._generate_next_piece()
        new_board = Board(board_width, board_height + 2)
        piece_loc = new_board.get_piece_start_loc(current_piece)
        queue = tuple((self._generate_next_piece() for i in range(queue_size)))
        move_history = tuple()
        self.update_gravity()
        self.data.initialize(current_piece, piece_loc, queue, new_board, move_history)

    def _find_farthest_left(self, piece, orientation, loc):
        farthest_left = loc

        while piece.can_move(self.data.board, farthest_left, orientation):
            farthest_left = farthest_left.get_adj(Adj_Grid_Names.W)

        return loc.c - farthest_left.c

    def _find_farthest_right(self, piece, orientation, loc):
        farthest_right = loc

        while piece.can_move(self.data.board, farthest_right, orientation):
            farthest_right = farthest_right.get_adj(Adj_Grid_Names.E)

        return farthest_right.c - loc.c
    
    def _find_farther_turn(self, piece, orientation, loc, rotation, depth):
        if depth == 0:
            return 0, (float("inf"), float("inf"))

        new_orientation = turn(orientation, rotation)
        if piece._can_rotate(self.data.board, loc, new_orientation, 0):
            farthest_left = self._find_farthest_left(piece, new_orientation, loc)
            farthest_right = self._find_farthest_right(piece, new_orientation, loc)
            can_rotate, (farthest_left1, farthest_right1) = self._find_farther_turn(piece, new_orientation, loc, rotation, depth - 1)
            return can_rotate + 1, (min(farthest_left, farthest_left1), min(farthest_right, farthest_right1))
        return 0, (0, 0)

    def _find_most_turns_left(self, piece, orientation, loc):
        return self._find_farther_turn(piece, orientation, loc, Controls.ROTATE_LEFT, 1)
    
    def _find_most_turns_right(self, piece, orientation, loc):
        return self._find_farther_turn(piece, orientation, loc, Controls.ROTATE_RIGHT,  2)

    def _get_rotation_moves(self, left, right):
        moves = []
        
        for i in range(left):
            moves.append((i, Controls.ROTATE_LEFT))

        for i in range(1, right):
            moves.append((i, Controls.ROTATE_RIGHT))

        return moves

    def _get_left_right_moves(self, left, right):
        moves = []

        for i in range(left):
            moves.append((i, Controls.MOVE_LEFT))

        for i in range(1, right):
            moves.append((i, Controls.MOVE_RIGHT))

        return moves

    def get_possible_moves(self, piece, piece_loc, piece_ori):
        # if piece_loc
        # TODO: add function to more easily add a variety of moves to items
        '''
            TODO: (order for rotation then move matters)
            check if can move left and right : can't if farthest left/right == 0 (this is the end of recursion)
            check if can rotate left or right : can't if farthest turn is 0 (180 == right * 2)
                check farthest move left and right for each orientation
                return (the number of turns, farther move: pos for right, neg for left; tuple if both)
            if number of turns is less than it's supposed to left = 1 and right = 2:
                then check secong arg then for moves >= that num, test again
            add the rotation you can do to the list
            add the moves you can do to the list (pass in second arg of rotations):
                when moves is greater than the second arg, rerun check rotate, and re-add if necessary
            check if there are caves (return height and farthest pertrusion):
                add soft drop
                recurse on each soft drop height
            add hard drop for each and make the thing a tuple
        '''
        piece = self.data.current_piece
        piece_loc = self.data.current_piece_loc
        orientation = piece.orientation
        move_left = self._find_farthest_left(piece, orientation, piece_loc)
        move_right = self._find_farthest_right(piece, orientation, piece_loc)

        rotate_right = self._find_most_turns_right(piece, orientation, piece_loc)
        rotate_left = self._find_most_turns_left(piece, orientation, piece_loc)

        if move_left == 0 and move_right == 0 and rotate_right[0] == 0 and rotate_left[0] == 0:
            return [[(1, None)]]

        moves = [[(1, Controls.HOLD)]]
        for rotation in self._get_rotation_moves(rotate_left[0], rotate_right[0]):
            for left_right in self._get_left_right_moves(move_left, move_right):
                moves.append([rotation, left_right])

        # TODO: add what to do with caves

        for move_set in moves:
            move_set.append((1, Controls.HARD_DROP))

        return moves

    def getLegalActions(self):
        '''
            TODO:
            - outputs (action, times to execute)
            - only recurs on softdrop
            - after recur, move left/right, and rotate left/right (with just varying numbers of executions)
            - maybe have the number of softdrop execution # =  heights of topology? (at least, soft drop height)
            - still needs to figure out how to consider a hard drop for all movements
            - returns a list of actions, each element contains (action, times to execute action)
        '''
        return self.get_possible_moves(self.data.current_piece, self.data.current_piece_loc, self.data.current_piece.orientation)

    # def getLegalActions(self):
    #     moves = [None]
    #     if self.data.gameover:
    #         return moves

    #     piece = self.data.current_piece.copy()
    #     pos = self.data.current_piece_loc.copy()
    #     orientation = piece.orientation
    #     board = self.data.board.deepCopy()

    #     if board._can_be_placed(piece, pos):
    #         moves.append(Controls.HARD_DROP)

    #     dirs = [Adj_Grid_Names.S, Adj_Grid_Names.W, Adj_Grid_Names.E]
    #     controls = [Controls.SOFT_DROP, Controls.MOVE_LEFT, Controls.MOVE_RIGHT]

    #     for dir, control in zip(dirs, controls):
    #         new_pos = pos.get_adj(dir)
    #         if board._can_be_placed(piece, new_pos) and not (piece, new_pos) in self.data.move_history:
    #             moves.append(control)

    #     rotations = [Controls.ROTATE_LEFT, Controls.ROTATE_RIGHT]
    #     for dir in rotations:
    #         copy = piece.copy()
    #         rotate = copy.rotate(board, pos, dir)
    #         if board._can_be_placed(copy, rotate)  and not (copy, rotate) in self.data.move_history:
    #             moves.append(dir)
        
    #     if not self.data.just_held:
    #         moves.append(Controls.HOLD)

    #     return moves

    def update_gravity(self):
        self.data.gravity = level_to_gravity(self.data.level)

        return self

    def get_gravity(self):
        return self.data.gravity

    def update_move_history(self):
        if len(self.data.move_history) < 5:
            self.data.move_history = self.data.move_history + ((self.data.current_piece.copy(), self.data.current_piece_loc.copy()),)
        else:
            self.data.move_history = self.data.move_history[1:] + ((self.data.current_piece.copy(), self.data.current_piece_loc.copy()),)
        return self

    def getSuccessor(self, action):
        successor = GameState(self)

        if action[0][1] is None:
            return successor

        if action[0][1] is Controls.PLACE:
            return successor.place_piece()

        for move in action:
            times, control  = move
            for i in range(times):
                successor = (successor.control_to_move[control](successor)).update_move_history()

        return successor

    # def getSuccessor(self, action):
    #     successor = GameState(self)

    #     if action is Controls.PLACE:
    #         return successor.place_piece()

    #     if action is None:
    #         return successor

    #     return (successor.control_to_move[action](successor)).update_move_history()
    
    def place_piece(self):
        self.data.board.place_piece(self.data.current_piece, self.data.current_piece_loc)
        self.data.just_held = False
        self.data.placed_blocks += 1
        self.data.gameover = self._check_gameover()
        self.data.can_place = False
        return self.clear_lines().level_up().update_gravity().next_piece()
    
    def _check_gameover(self):
        board_list = self.data.board.asList()

        top_row = board_list[-1]
        for c in range(len(board_list[0])):
            if top_row[c] != Board_View.Board_Values.EMPTY:
                return True
            
        return False

    def get_queue_size(self):
        return len(self.data.queue)

    def hard_drop(self):
        final_pos, blocks_dropped = self.data.current_piece.hard_drop(self.data.board, self.data.current_piece_loc)
        self.data.current_piece_loc = final_pos
        self.data.score += blocks_dropped * 2
        self.data.can_place = True
        return self
    
    def check_can_place(self):
        return self.data.can_place
    
    def _move_dir(self, direction):
        old_loc = self.data.current_piece_loc
        new_loc = old_loc.get_adj(direction)
        if self.data.current_piece.can_move(self.data.board, new_loc):
            self.data.current_piece_loc = new_loc

            if direction == Adj_Grid_Names.S:
                self.data.score += 1

        return self

    def _rotate(self, rotation):
        self.data.current_piece_loc = self.data.current_piece.rotate(self.data.board, self.data.current_piece_loc, rotation)
        
        return self

    def _get_idle_move(self):
        return self.data.current_piece_loc.get_adj(Adj_Grid_Names.S)

    def _move_idley(self):
        if not self.is_piece_set():
            self.data.current_piece_loc = self._get_idle_move()
        else:
            self.data.can_place = True
        return self

    def is_piece_set(self):
        return not self.data.current_piece.can_move(self.data.board, self._get_idle_move())

    def get_level(self):
        return self.data.level
    
    def deepCopy(self):
        return GameState(self)
    
    def shallowCopy(self):
        copy = GameState(self)
        copy.data = self.data.shallowCopy()
        return copy    
    
    def getPieceLocation(self):
        return self.data.current_piece_loc.copy()
    
    def getPiece(self):
        return self.data.current_piece.copy()
    
    def getPieceOrientation(self):
        return self.data.current_piece.orientation

    def next_piece(self):
        new_piece = self.data.queue[0]
        new_loc = self.data.board.get_piece_start_loc(self.data.current_piece)
        new_queue = self.update_queue()
        self.data.initialize(new_piece, new_loc, new_queue)

        return self

    def hold(self):
        if self.data.hold:
            temp = self.data.hold
            self.data.hold = self.data.current_piece
            new_loc = self.data.board.get_piece_start_loc(temp)
            self.data.initialize(temp, new_loc, self.data.queue)
        else:
            self.data.hold = self.data.current_piece
            self.next_piece()

        self.data.just_held = True

        return self
    
    def clear_lines(self):
        num_cleared = self.data.board.clear_lines()
        if num_cleared == 0:
            self.data.comboCount = 0
        else:
            self.data.score += (num_cleared * 200 - 100 if num_cleared < 4 else 800) * self.data.level
            self.data.comboCount += 1
            self.data.lines_cleared_on_lvl += num_cleared

        return self
    
    def level_up(self):
        if self.data.lines_cleared_on_lvl >= self.data.lines_to_clear_on_lvl:
            self.data.lines_cleared_on_lvl -= self.data.lines_to_clear_on_lvl
            self.data.level += 1
            self.data.lines_to_clear_on_lvl = get_level_line_clears(self.data.level)
            
        return self

    def update_queue(self):
        next_piece_in_queue = self._generate_next_piece()

        while next_piece_in_queue in self.data.queue and random() < 0.05:
            next_piece_in_queue = self._generate_next_piece()

        return self.data.queue[1:] + (next_piece_in_queue,)

    def _generate_next_piece(self):
        piece = randint(0, len(GAME_PIECES) - 1)
        return GAME_PIECES[piece]()    

    def is_game_over(self):
        return self.data.gameover

    def deepCopy(self):
        return GameState(self)
    
    def get_board(self):
        return self.data.board.deepCopy()

    def get_gui_board(self):
        return self.data.board.asList(self.data.current_piece, self.data.current_piece_loc)
    
    def get_board_size(self):
        board = self.data.board
        return board.get_width(), board.get_height() - 2
    
    def get_score(self):
        return self.data.score
    
    def get_piece(self):
        return self.data.current_piece.copy()
    
    def get_piece_loc(self):
        return self.data.current_piece_loc.copy()
    
    def get_combo_count(self):
        return self.data.comboCount
    
    def get_queue(self):
        return tuple((piece.copy() for piece in self.data.queue))

    def get_lines_till_next_level(self):
        return self.data.lines_to_clear_on_lvl - self.data.lines_cleared_on_lvl
    
    def get_hold(self):
        hold = self.data.hold
        return None if hold is None else hold.copy()
    
    def get_num_blocks_placed(self):
        return self.data.placed_blocks
    
    def __hash__(self) -> int:
        return hash(self.data)