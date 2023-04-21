# keyboardAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from tetris_game import Agent, Controls
import random
from time import time

class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """
    HOLD_KEY = 'Up'
    ROTATE_LEFT_KEY = 'Left'
    ROTATE_RIGHT_KEY = 'Right'
    MOVE_LEFT_KEY = 'a'
    MOVE_RIGHT_KEY = 'd'
    SOFT_DROP_KEY = 's'
    HARD_DROP_KEY = 'w'

    def __init__( self):

        self.keys = []
        self.click_time = 0.03
        self.last_click_time = time()
        self.hard_drop_cool = False
        self.rotateL_drop_cool = False
        self.rotateR_drop_cool = False

    def getAction( self, state, agents):
        from graphicsUtils import keys_waiting
        from graphicsUtils import keys_pressed
        self.keys = list(keys_waiting()) + list(keys_pressed())

        legal = state.getLegalActions()
        move = self.getMove(legal)

        return move

    def getMove(self, legal):
        current_time = time()
        move = None
        if current_time - self.last_click_time >= self.click_time:
            self.last_click_time = current_time
            keys = set(self.keys)

            if self.hard_drop_cool and not self.HARD_DROP_KEY in keys:
                self.hard_drop_cool = False

            if self.rotateL_drop_cool and not self.ROTATE_LEFT_KEY in keys:
                self.rotateL_drop_cool = False

            if self.rotateR_drop_cool and not self.ROTATE_RIGHT_KEY in keys:
                self.rotateR_drop_cool = False

            legal_moves = legal
            if   (self.MOVE_LEFT_KEY in keys) and Controls.MOVE_LEFT in legal_moves:  move = Controls.MOVE_LEFT
            if   (self.MOVE_RIGHT_KEY in keys) and Controls.MOVE_RIGHT in legal_moves:  move = Controls.MOVE_RIGHT
            if   (self.HOLD_KEY in keys) and Controls.HOLD in legal_moves:  move = Controls.HOLD
            if   (self.HARD_DROP_KEY in keys) and Controls.HARD_DROP in legal_moves and not self.hard_drop_cool:  
                move = Controls.HARD_DROP 
                self.hard_drop_cool = True
            if   (self.ROTATE_LEFT_KEY in keys) and Controls.ROTATE_LEFT in legal_moves and not self.rotateL_drop_cool:  
                move = Controls.ROTATE_LEFT
                self.rotateL_drop_cool = True
            if   (self.ROTATE_RIGHT_KEY in keys) and Controls.ROTATE_RIGHT in legal_moves and not self.rotateR_drop_cool: 
                move = Controls.ROTATE_RIGHT
                self.rotateR_drop_cool = True
            if   (self.SOFT_DROP_KEY in keys) and Controls.SOFT_DROP in legal_moves:  
                move = Controls.SOFT_DROP
        return move

class KeyboardAgent2(KeyboardAgent):
    """
    A second agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    HOLD_KEY = 'c'
    ROTATE_LEFT_KEY = 'z'
    ROTATE_RIGHT_KEY = 'Up'
    MOVE_LEFT_KEY = 'Left'
    MOVE_RIGHT_KEY = 'Right'
    SOFT_DROP_KEY = 'Down'
    HARD_DROP_KEY = 'Space'
