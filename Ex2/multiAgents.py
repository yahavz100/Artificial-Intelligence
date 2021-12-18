# Yahav Zarfati

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


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        minDistance: int = 2
        MINUS: int = -1

        # Calculate the distances from next pacman position to each next ghost position, using manhattan distance
        ghostDistance: list = []
        for ghost in newGhostStates:
            distance: int = manhattanDistance(newPos, ghost.configuration.getPosition())

            # If distance is lower than minimal step, return infinity
            if distance < minDistance:
                return float('-inf')
            ghostDistance.append(distance)

        # Calculate the distances from each food current location to next pacman position
        foodDistanceList: list = []
        foodList: list = currentGameState.getFood().asList()
        for food in foodList:
            # Multiply by -1 in order to return higher score for better choices
            foodDistanceList.append(MINUS * manhattanDistance(food, newPos))

        return max(foodDistanceList)


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        # For each legal action evaluate all tree of actions using MinMax and return the best scenario
        actionsDict = {}
        legalActions = gameState.getLegalActions()

        for action in legalActions:
            # Execute minMax on pacman successors for each action, starting depth = 0 and opponent agent
            value = self.minMaxValue(gameState.generateSuccessor(0, action), 0, 1)
            actionsDict[action] = value

        # Find the action with max value and return it
        actionValue = max(actionsDict.values())
        for act, actVal in actionsDict.items():
            # If found max value return its key
            if actVal == actionValue:
                print(actVal)
                return act

    def minMaxValue(self, gameState, depth, agent):
        # If reached to all agents(pacman and ghosts) go deeper in the tree and start from 1st agent(pacman)
        if agent == gameState.getNumAgents():
            depth += 1
            agent = 0

        # Check if reached end of the tree(leaf)\depth or win\lose return its value
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actionsDict = {}
        legalActions = gameState.getLegalActions(agent)

        nextAgent = agent + 1
        # For each type of agent(pacman/ghost) calculate its minMax value using his successors(each action)
        # If its pacman return max value
        if agent == 0:
            for action in legalActions:
                value = self.minMaxValue(gameState.generateSuccessor(agent, action), depth, nextAgent)
                actionsDict[action] = value
            return max(actionsDict.values())

        # Otherwise, its a ghost, return min value
        else:
            for action in legalActions:
                value = self.minMaxValue(gameState.generateSuccessor(agent, action), depth, nextAgent)
                actionsDict[action] = value
            return min(actionsDict.values())


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actionsDict = {}
        alpha = float('-inf')
        beta = float('inf')
        value = float('-inf')

        # For each legal action evaluate all tree of actions using MinMax with AB pruning and return the best scenario
        legalActions = gameState.getLegalActions()
        for action in legalActions:
            # Execute MinMaxAB pruning on pacman successors(ghosts) for each action, starting depth = 0
            # and opponent agent
            value = max(self.alphaBetaPruning(gameState.generateSuccessor(0, action), 0, 1, alpha, beta), value)

            if value > beta:
                return value

            alpha = max(alpha, value)
            actionsDict[action] = value

        # Find the action with max value and return it
        actionValue = max(actionsDict.values())
        for act, actVal in actionsDict.items():
            # If found max value return its key
            if actVal == actionValue:
                return act

    def alphaBetaPruning(self, gameState, depth, agent, alpha, beta):
        # If reached to all agents(pacman and ghosts) go deeper in the tree and start from 1st agent(pacman)
        if agent == gameState.getNumAgents():
            depth += 1
            agent = 0

        # Check if reached end of the tree(leaf)\depth or win\lose return its value
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agent)

        nextAgent = agent + 1
        # For each type of agent(pacman/ghost) calculate its minMax value using his successors(each action)
        # If its pacman return max value
        if agent == 0:
            # Implement AlphaBeta pruning
            value = float('-inf')
            for action in legalActions:
                value = max(self.alphaBetaPruning(gameState.generateSuccessor(agent, action),
                                                  depth, nextAgent, alpha, beta), value)
                if value > beta:
                    return value

                alpha = max(alpha, value)
            return value

        # Otherwise, its a ghost, return min value
        else:
            # Implement AlphaBeta pruning
            value = float('inf')
            for action in legalActions:
                value = min(self.alphaBetaPruning(gameState.generateSuccessor(agent, action),
                                                  depth, nextAgent, alpha, beta), value)
                if value < alpha:
                    return value

                beta = min(beta, value)
            return value
