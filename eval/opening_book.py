import copy
import numpy as np
import pickle
import argparse
from random import shuffle, seed
from tqdm import trange
from collections import defaultdict

X = "X"
O = "O"
E = None

convert = {
    X: -1,
    E: 0,
    O: 1
}

inv_convert = {
    -1: X,
    0: E,
    1: O
}

opening_book = {}

FIRST_PLAYER = X

def key(board, player_move):
    flat_board = [convert[player_move]]
    for c in range(3):
        for r in range(3):
            flat_board.append(convert[board[c][r]])
    return np.array(flat_board, dtype=np.int64).data.tobytes()

def inv_key(key):
    array = np.frombuffer(key, dtype=np.int64)
    player_move = inv_convert[array[0]]
    board = [inv_convert[e] for e in array[1:]]
    board = [board[0:3], board[3:6], board[6:9]]
    return board, player_move


def initial_state():
    return [[E, E, E],
            [E, E, E],
            [E, E, E]]


def player(board):
    global FIRST_PLAYER
    count = sum(row.count(X) + row.count(O) for row in board)
    if FIRST_PLAYER == X:
        return O if count % 2 != 0 else X
    else:
        return X if count % 2 != 0 else O


def actions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == E]


def result(board, action):
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    for player in (X, O):
        # Check rows and columns
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
                return player
        # Check diagonals
        if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
            return player
    return None


def terminal(board):
    return winner(board) is not None or all(cell != E for row in board for cell in row)


def utility(board):
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    opening_book = {}
    def max_value(board):
        if terminal(board):
            return utility(board)
        v = float('-inf')
        best_action = None
        for action in actions(board):
            min_val = min_value(result(board, action))
            if min_val > v:
                v = min_val
                best_action = action
        current_player = player(board)
        opening_book[key(board, current_player)] = best_action
        return v

    def min_value(board):
        if terminal(board):
            return utility(board)
        v = float('inf')
        best_action = None
        for action in actions(board):
            max_val = max_value(result(board, action))
            if max_val < v:
                v = max_val
                best_action = action

        current_player = player(board)
        opening_book[key(board, current_player)] = best_action
        return v

    current_player = player(board)

    if current_player == X:
        return max(actions(board), key=lambda a: min_value(result(board, a))), opening_book
    else:
        action = min(actions(board), key=lambda a: max_value(result(board, a)))
        return action, opening_book

def print_board(board):
    for row in board:
        print(row)
    print('')


def opening_book_response(board, p):
    action = opening_book[key(board, p)]
    return result(board, action)


def minmax_response(board):
    action = minimax(board)
    return result(board, action)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--make_book', action='store_true')
    args = parser.parse_args()

    if args.make_book:

        opening_book = {}

        for first in [X, O]:
            FIRST_PLAYER = first
            board = initial_state()
            action, book = minimax(board)
            book[key(initial_state(), first)] = (1, 1)
            opening_book.update(book)

        with open('opening_book.pkl', 'wb') as f:
            pickle.dump(opening_book, f)

    with open('opening_book.pkl', 'rb') as f:
        opening_book = pickle.load(f)

        for k in opening_book:
            board, p = inv_key(k)
            a = opening_book[k]
            assert a in set(actions(board))

        print('symmetric book')
        FIRST_PLAYER = X
        test_board = [['X', None, None], [None, 'X', None], [None, 'O', 'O']]
        print_board(test_board)
        action = opening_book[key(test_board, X)]
        print(action)
        print_board(opening_book_response(test_board, X))

        print('symmetric book')
        FIRST_PLAYER = O
        test_board = [['O', None, None], [None, 'O', None], [None, X, X]]
        print_board(test_board)
        action = opening_book[key(test_board, O)]
        print(action)
        print_board(opening_book_response(test_board, O))


    board = [[E, E, E],
             [E, X, E],
             [E, E, O]]

    k = key(board, X)
    assert inv_key(k)[0] == board
    assert inv_key(k)[1] == X

    for k in opening_book:
        assert k == key(*inv_key(k))

    # print()
    # for row in board:
    #     print(row)
    #
    #
    # board = [[E, E, E],
    #          [E, X, E],
    #          [E, E, E]]
    #
    # for row in board:
    #     print(row)
    # action = opening_book[key(board)]
    # board = result(board, action)
    # print(action)
    # for row in board:
    #     print(row)

    # play against random

    check = 'draw'

    seed(1)

    # player goes first
    for first in [X, O]:
        FIRST_PLAYER = first
        winner_table = defaultdict(int)
        for i in trange(1000):
            board = initial_state()
            while True:
                # print_board(board)
                valid_actions = actions(board)
                shuffle(valid_actions)
                action = valid_actions[0]
                board = result(board, action)
                if terminal(board):
                    break

                try:
                    action = opening_book[key(board, player(board))]
                except KeyError:
                    print_board(board)
                    check = 'failed'
                    break

                # action = minimax(board)
                check = 'draw'
                # print_board(board)
                board = result(board, action)
                if terminal(board):
                    break

            w = winner(board)
            w = w if w is not None else check
            winner_table[w] += 1
            # print(w)
            # print_board(board)

        print(winner_table)

    # ai goes first
    for first in [X, O]:
        FIRST_PLAYER = first
        winner_table = defaultdict(int)

        for i in range(1000):
            trajectory = []
            board = initial_state()
            while True:
                # print_board(board)
                try:
                    action = opening_book[key(board, player(board))]
                except KeyError:
                    print_board(board)
                    check = 'failed'
                    break

                # action = minimax(board)
                check = 'draw'
                # print_board(board)
                board = result(board, action)

                trajectory.append(board)
                if terminal(board):
                    break

                valid_actions = actions(board)
                shuffle(valid_actions)
                action = valid_actions[0]
                board = result(board, action)
                trajectory.append(board)
                if terminal(board):
                    break

            w = winner(board)
            w = w if w is not None else check
            winner_table[w] += 1

            # if w == O and x_first is True:
            #     for b in trajectory:
            #         print_board(b)
            #     print(w)
            #     print_board(board)
        print(f'ai first first{first}')
        print(winner_table)