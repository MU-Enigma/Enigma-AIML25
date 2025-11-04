import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1, v2):
    dp = dot_product(v1, v2)
    mag1 = math.sqrt(dot_product(v1, v1))
    mag2 = math.sqrt(dot_product(v2, v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dp / (mag1 * mag2)

def normalize_l1(vec):
    s = sum(abs(x) for x in vec)
    return [x / s for x in vec] if s != 0 else vec

def normalize_l2(vec):
    s = math.sqrt(sum(x**2 for x in vec))
    return [x / s for x in vec] if s != 0 else vec

def normalize_minmax(vec):
    mn, mx = min(vec), max(vec)
    if mx == mn:
        return [0 for _ in vec]
    return [(x - mn) / (mx - mn) for x in vec]

def greedy_move_tictactoe(board, player):
    opponent = 'O' if player == 'X' else 'X'

    def is_winning_move(b, r, c, mark):
        b[r][c] = mark
        win = any(all(b[i][j] == mark for j in range(3)) for i in range(3)) or \
              any(all(b[i][j] == mark for i in range(3)) for j in range(3)) or \
              all(b[i][i] == mark for i in range(3)) or \
              all(b[i][2 - i] == mark for i in range(3))
        b[r][c] = ' '
        return win

    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ' and is_winning_move(board, r, c, player):
                return r, c

    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ' and is_winning_move(board, r, c, opponent):
                return r, c

    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ':
                return r, c

    return None
