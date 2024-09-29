import copy
import pickle
import argparse
from random import shuffle, seed
from collections import defaultdict


X = 1
O = 0
E = -1

# convert = {
#     X: 0,
#     E: -1,
#     O: 1
# }
#
# inv_convert = {
#     0: X,
#     -1: E,
#     1: O
# }

def flatten(board):
    flat_board = []
    for c in range(3):
        for r in range(3):
            flat_board.append(board[c][r])
    return flat_board

def unflatten(flat_board):
    board = [[-1]*3 for i in range(3)]
    for i in range(9):
        c, r = i // 3, i % 3
        board[c][r] = flat_board[i]
    return board


def tictactoe_hash(board, player):
    """
    Perfect hash function for a tic-tac-toe board state.

    Args:
    board (list): A list of 9 elements representing the board state.
                  Each element can be 'X', 'O', or ' ' (empty).

    Returns:
    int: A unique hash value for the given board state.
    """
    # Define a mapping for each possible cell state
    state_map = {E: 0, X: 1, O: 2}
    board = flatten(board)

    # Convert the board to a base-3 number
    hash_value = 0
    for i, cell in enumerate(board):
        hash_value += state_map[cell] * (3 ** i)

    hash_value += (state_map[player]-1) * (3 ** 9)

    return hash_value


def inverse_tictactoe_hash(hash_value):
    """
    Inverse function for tictactoe_hash.

    Args:
    hash_value (int): The hash value of a tic-tac-toe board state.

    Returns:
    list: A list of 9 elements representing the board state.
          Each element is Cell.E, Cell.X, or Cell.O.
    """
    inverse_state_map = {0: E, 1: X, 2: O}
    board = []

    for i in range(9):
        cell_value = hash_value % 3
        board.append(inverse_state_map[cell_value])
        hash_value //= 3

    board = unflatten(board)

    player = hash_value % 3 + 1
    player = inverse_state_map[player]
    hash_value //= 3

    return board, player


def key(board, player):
    return tictactoe_hash(board, player)

def inv_key(hash_value):
    return inverse_tictactoe_hash(hash_value)


opening_book = []

FIRST_PLAYER = X

# def key(board, player_move):
#     flat_board = [convert[player_move]]
#     for c in range(3):
#         for r in range(3):
#             flat_board.append(convert[board[c][r]])
#     return np.array(flat_board, dtype=np.int64).data.tobytes()
#
# def inv_key(key):
#     array = np.frombuffer(key, dtype=np.int64)
#     player_move = inv_convert[array[0]]
#     board = [inv_convert[e] for e in array[1:]]
#     board = [board[0:3], board[3:6], board[6:9]]
#     return board, player_move


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
    opening_book = new_book()
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
        action = max(actions(board), key=lambda a: min_value(result(board, a)))
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

MAX_ENTRIES=3**9*2

def new_book():
    return [(-1, -1)] * MAX_ENTRIES

# since we are using a perfect hash merging books is simple
def merge(book1, book2):
    for i, action in enumerate(book2):
        if action != (-1, -1):
            book1[i] = book2[i]
    return book1


def make_book():
    global FIRST_PLAYER
    opening_book = new_book()

    for first in [X, O]:
        FIRST_PLAYER = first
        board = initial_state()
        action, book = minimax(board)
        book[key(initial_state(), first)] = (1, 1)
        opening_book = merge(opening_book, book)

    with open('opening_book.pkl', 'wb') as f:
        pickle.dump(opening_book, f)

    remmaped_action_book = new_book()
    for k, (c, r) in enumerate(opening_book):
        remmaped_action_book[k] = c * 3 + r

    with open('tictactoe_book.py', 'w') as f:
        python_code = f'action_book = {remmaped_action_book}'
        f.write(python_code)



# with open('tictactoe_book.py', 'r') as f:
#     action_book = json.load(f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--make_book', action='store_true')
    args = parser.parse_args()

    board = [[E, E, E],
             [E, X, E],
             [E, E, O]]


    flat_board = flatten(board)
    unflat_board = unflatten(flat_board)
    assert board == unflatten(flatten(board))

    k = key(board, X)
    decoded_board, decoded_player= inv_key(k)
    assert decoded_board == board
    assert decoded_player == X

    k = key(board, O)
    decoded_board, decoded_player = inv_key(k)
    assert decoded_board == board
    assert decoded_player == O

    if args.make_book:
        make_book()

    with open('opening_book.pkl', 'rb') as f:
        opening_book = pickle.load(f)

        for k, _ in enumerate(opening_book):
            assert k == key(*inv_key(k))

        for k, _ in enumerate(opening_book):
            board, p = inv_key(k)
            a = opening_book[k]
            if a != (-1, -1):
                assert a in set(actions(board)), f"{a}"

        print('X to move')
        FIRST_PLAYER = X
        test_board = [[X, E, E],
                      [E, X, E],
                      [E, O, O]]
        print_board(test_board)
        action = opening_book[key(test_board, X)]
        print(action)
        assert action == (2, 0), f"expected (0, 2) got {action}"
        print_board(opening_book_response(test_board, X))

        print('O to move')
        FIRST_PLAYER = O
        test_board = [[X, E, E], [E, X, E], [E, O, O]]
        print_board(test_board)
        action = opening_book[key(test_board, O)]
        print(action)
        assert action == (0, 2)
        print_board(opening_book_response(test_board, O))

        print('X to move')
        FIRST_PLAYER = O
        test_board = [[E, E, E], [E, X, E], [E, E, E]]
        print_board(test_board)
        hash_code = key(test_board, O)
        action = opening_book[hash_code]
        print(f'hash {hash_code}, action {action}')
        assert action == (0, 0)
        print_board(opening_book_response(test_board, O))

    board = [[E, E, E],
             [E, E, E],
             [E, E, E]]

    print(key(board, 0))
    action = opening_book[key(board, 0)]
    assert action == (1, 1)

    board = [[E, E, E],
             [E, 0, E],
             [E, E, E]]

    print(key(board, 1))
    action = opening_book[key(board, 1)]
    print(action)

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
        for i in range(1000):
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


