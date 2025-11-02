import math
import numpy

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ReLU function 
def relu(x):
    return max(0, x)

# tanh function 
def tanh(x):
    return math.tanh(x)

#vector dot product
def vector_dot_product(x,y):
    return x@y

#cosine similarity
def simple_heuristic(board, player):
    opponent = 'O' if player == 'X' else 'X'
    wins = [
        [0,1,2],[3,4,5],[6,7,8], # rows
        [0,3,6],[1,4,7],[2,5,8], # columns
        [0,4,8],[2,4,6]          # diagonals
    ]

    score = 0

    for pos in wins:
        vals = [board[i] for i in pos]
        # if only the player is in that line → good
        if opponent not in vals:
            score += vals.count(player)
        # if opponent controls that line → bad
        elif player not in vals:
            score -= vals.count(opponent)
    
    return score

#Heuristic function
''' This heuristic measures how far each tile is from its goal position (up/down/left/right steps).
    The smaller the total distance → the closer the grid is to the goal.
    Still very easy to understand, just one level deeper than counting mismatches.'''
def heuristic(current, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            for x in range(3):
                for y in range(3):
                    if current[i][j] == goal[x][y]:
                        distance += abs(i - x) + abs(j - y)
    return distance
