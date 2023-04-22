from enum import IntEnum
import heapq

class Adj_Grid_Names(IntEnum):
    N_W = 0
    N = 1
    N_E = 2
    W = 3
    CENTER = 4
    E = 5
    S_W = 6
    S = 7
    S_E = 8

class Grid:
    adj_grids = {Adj_Grid_Names.N_W: lambda grid: grid.shift((-1, 1)),
                 Adj_Grid_Names.N: lambda grid: grid.shift((0, 1)),
                 Adj_Grid_Names.N_E: lambda grid: grid.shift((1, 1)),
                 Adj_Grid_Names.W: lambda grid: grid.shift((-1, 0)),
                 Adj_Grid_Names.CENTER: lambda grid: grid.shift((0, 0)),
                 Adj_Grid_Names.E: lambda grid: grid.shift((1, 0)),
                 Adj_Grid_Names.S_W: lambda grid: grid.shift((-1, -1)),
                 Adj_Grid_Names.S: lambda grid: grid.shift((0, -1)),
                 Adj_Grid_Names.S_E: lambda grid: grid.shift((1, -1))}

    def __init__(self, c, r = None) -> None:
        if r is None:
            self.c, self.r = c
        else:
            self.c = c
            self.r = r

    '''
    Returns the value in the format of a 2-d array (c, r)
    '''
    def get_points(self):
        return self.c, self.r
    
    '''
    Returns a new grid shifted over by d_r, d_c from self's center
    '''
    def shift(self, d_c, d_r = None):
        if d_r is None:
            d_c, d_r = d_c
        return Grid(self.c + d_c, self.r + d_r)

    def get_adj(self, *adj: Adj_Grid_Names):
        new_grid = self
        for adjacent in adj:
            new_grid = Grid.adj_grids[adjacent](new_grid)
        return new_grid
    
    def __sub__(self, other):
        return Vector(self.shift(-other.c, -other.r))
    
    def __add__(self, other):
        return Vector(self.shift(other.c, other.r))
    
    def copy(self):
        return Grid(self.c, self.r)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Grid) and __value.get_points() == self.get_points()
    def __hash__(self) -> int:
        return hash((self.c, self.r))
    
class Vector(Grid):
    def __init__(self, *values):
        num_values = len(values)

        if num_values == 1:
            super().__init__(values[0].c, values[0].r)
        elif num_values == 2:
            super().__init__(values[0], values[1])
        else:
            '''
            Acts on a series of lambdas to create a new shift vector
            '''

            new_grid = Grid(0, 0)

            for grid in values:
                new_grid = new_grid.get_adj(values)

            self.__init__(new_grid.c, new_grid.r)
        
    def __truediv__(self, factor: int):
        return Vector(self.c/factor, self.r/factor)

    def copy(self):
        return Vector(self.c, self.r)
        

def shift_blocks(grid_list, vector: Vector):
    shifted = list()

    for grid in grid_list:
        shifted.append(grid + vector)

    return tuple(shifted)


def lookup(name, namespace):
    """
    Get a method or class from any imported module from its name.
    Usage: lookup(functionName, globals())
    """
    dots = name.count('.')
    if dots > 0:
        moduleName, objName = '.'.join(
            name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(moduleName)
        return getattr(module, objName)
    else:
        modules = [obj for obj in list(namespace.values()) if str(
            type(obj)) == "<type 'module'>"]
        options = [getattr(module, name)
                   for module in modules if name in dir(module)]
        options += [obj[1]
                    for obj in list(namespace.items()) if obj[0] == name]
        if len(options) == 1:
            return options[0]
        if len(options) > 1:
            raise Exception('Name conflict for %s')
        raise Exception('%s not found as a method or class' % name)
    
"""
 Data structures useful for implementing SearchAgents
"""

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))