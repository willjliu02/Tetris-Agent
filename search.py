import util

def updatePathCosts(current_state, successor, locCost):
    child, cost = successor
    tempCost = cost + locCost[current_state]
    if not child in locCost or tempCost < locCost[child]:
        locCost[child] = tempCost
        return True

    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    explored = set()
    toExplore = util.PriorityQueue()

    state = problem.getStartState()
    path = []

    forks = {state: path}
    locCost = {state: 0}

    while not problem.isGoalState(state):
        explored.add(state)

        for child, action, cost in problem.getSuccessors(state):
            if not child in explored:
                if updatePathCosts(state, (child, cost), locCost):
                    this_path = path.copy()
                    this_path.append(action)
                    forks[child] = this_path

                toExplore.update(child, locCost[child])
                    
        child = toExplore.pop()
        path = forks[child]
        state = child

    return path