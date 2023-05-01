# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foodDistances) > 0:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 0

        # Calculate the distance to the nearest unscared ghost
        unscaredGhostDistances = []
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            if scaredTime == 0:
                unscaredGhostDistances.append(manhattanDistance(newPos, ghostState.getPosition()))
        if len(unscaredGhostDistances) > 0:
            minUnscaredGhostDistance = min(unscaredGhostDistances)
        else:
            minUnscaredGhostDistance = float("inf")

        # Calculate the total score
        score = successorGameState.getScore()

        # Add penalty for being close to unscared ghosts
        if minUnscaredGhostDistance < 3:
            score -= 1000

        # Add bonus for being close to food
        score += 10 / (minFoodDistance + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            value = float("-inf")
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v, _ = minValue(successor, depth, agentIndex + 1)
                if v > value:
                    value = v
                    bestAction = action
            return value, bestAction

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            value = float("inf")
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v, _ = maxValue(successor, depth - 1, 0)
                else:
                    v, _ = minValue(successor, depth, agentIndex + 1)
                if v < value:
                    value = v
                    bestAction = action
            return value, bestAction

        _, action = maxValue(gameState, self.depth, 0)
        return action
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxValue(state: GameState, depth: int, alpha: float, beta: float) -> float:
            v = -float("inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v = max(v, value(successor, depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> float:
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, value(successor, depth - 1, 0, alpha, beta))
                else:
                    v = min(v, value(successor, depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(state: GameState, depth: int, agentIndex: int, alpha: float, beta: float) -> float:
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth, alpha, beta)
            else:
                return minValue(state, depth, agentIndex, alpha, beta)

        alpha = -float("inf")
        beta = float("inf")
        bestAction = Directions.STOP
        v = -float("inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, self.depth, 1, alpha, beta)
            if score > v:
                v = score
                bestAction = action
            if v > beta:
                return bestAction
            alpha = max(alpha, v)
        return bestAction
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state: GameState, agent_index: int, depth: int) -> float:
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legal_actions: List[str] = state.getLegalActions(agent_index)
            num_legal_actions: int = len(legal_actions)

            if agent_index == 0:  # Pacman's turn
                best_score: float = float('-inf')
                for action in legal_actions:
                    successor_state: GameState = state.generateSuccessor(agent_index, action)
                    score: float = expectimax(successor_state, agent_index + 1, depth)
                    best_score = max(best_score, score)
                return best_score

            else:  # Ghosts' turn
                total_score: float = 0
                for action in legal_actions:
                    successor_state: GameState = state.generateSuccessor(agent_index, action)
                    if agent_index == state.getNumAgents() - 1:
                        total_score += expectimax(successor_state, 0, depth + 1)  # next turn is Pacman's
                    else:
                        total_score += expectimax(successor_state, agent_index + 1, depth)
                return total_score / num_legal_actions

        legal_actions: List[str] = gameState.getLegalActions(0)
        num_legal_actions: int = len(legal_actions)
        best_action: str = Directions.STOP
        best_score: float = float('-inf')

        for action in legal_actions:
            successor_state: GameState = gameState.generateSuccessor(0, action)
            score: float = expectimax(successor_state, 1, 0)  # 1 = first ghost, 0 = initial depth
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Based on the function from problem 1 which is reflex agent evaluation function,
    I have made the function to prioritize both distance to the nearest food and nearest unscared ghost.
    Then, the pacman is required to finish all the remaining foods while trying to stay away from the ghosts.
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the nearest food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodGrid.asList()]
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
    else:
        minFoodDistance = 0

    # Calculate the distance to the nearest unscared ghost
    unscaredGhostDistances = []
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        if scaredTime == 0:
            unscaredGhostDistances.append(manhattanDistance(pacmanPos, ghostState.getPosition()))
    if len(unscaredGhostDistances) > 0:
        minUnscaredGhostDistance = min(unscaredGhostDistances)
    else:
        minUnscaredGhostDistance = float("inf")

    # Calculate the total score
    score = currentGameState.getScore()

    # Add bonus for being close to food
    score += 10 / (minFoodDistance + 1)

    # Add bonus for remaining capsules
    # score += 100 * len(capsuleList)

    # Add bonus for remaining food
    score += 1000 / (len(foodGrid.asList()) + 1)

    # Add penalty for being close to unscared ghosts
    if minUnscaredGhostDistance < 2:
        score -= 1000

    return score
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
