# Artificial-Intelligence
- Task 1 - Pacman - Search Problems.
- Task 2 - Pacman - Multi agents, minimax and alpha-beta pruning.
- Task 3 - Movie recommendation system, analyze data, collaborative filtering and precision measuring.

## Task 1 - Pacman
Demonstrating how to solve problems by searching as part of Artificial Intelligence course at Bar Ilan University.<br/>
The main goal of this exercise is to define and implement searching problems, solve them using uninformed search and heuristic search.

There is a list of commands(commands.txt) initiating different mazes with different goals using a number of algorithms:
1. Implemented DFS(Graph Search) using Stack.
2. Implemented BFS(Graph Search) using Queue.
3. Implemented Uniform Cost Search using Priority Queue.
4. Implemented A* Search using Priority Queue.

Defined Corners-Problem search problem: State, Goal and Successors.

Implemented Heuristic function(admissible and consistent) for Corners-Problem and Food Search-Problem.<br/>
Function calculates Manhattan distances from current position to closest goal in each state.

![alt text](Ex1/Extra/pac-astrar-seq.gif)

![alt text](Ex1/Extra/Capture.jpg)

## Task 2 - Pacman multi agents
Implemented number of agents - minimax, alpha-beta pruning etc.
1. Improving the Reflex agent using game state information.
2. Adverserial search agent using Minimax.
3. Implement alpha-beta pruning agent in order to explore more efficiently the minimax tree.

![alt text](Ex2/Extra/pac-alphabeta-seq.gif)

## Task 3 - Movie recommendation system
The system is built from 3 parts:
1. Analyzing the data.
2. Creating a recommendation based on collaborative filtering: user based, item based.
3. Evaluate the system precision using: P@10, ARHA, RMSE.

![alt text](Ex3/Extra/capture1.jpg)

The prediction matrix is built using cosine similarity metric.
![alt text](Ex3/Extra/capture2.jpg) <br/>
![alt text](Ex3/Extra/capture3.jpg) <br/>
![alt text](Ex3/Extra/capture4.jpg) <br/>
![alt text](Ex3/Extra/capture5.jpg) <br/>
