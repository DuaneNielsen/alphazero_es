import pgx.tic_tac_toe
from pgx import core
from jax import Array
import jax.numpy as jnp
from pgx.tic_tac_toe import _win_check
from tictactoe_book import action_book

action_book = jnp.array(action_book)
base3_hash = jnp.array([3 ** i for i in range(10)])


def hash_state(state: core.State):
    player = state.current_player[jnp.newaxis] + 1  # pgx has the wierd player convention
    board = state._board + 1  # the board uses -1 based indexing
    hashable = jnp.concatenate([board, player])
    hashable *= base3_hash
    hashcode = hashable.sum(-1)
    return hashcode


def _step(state: core.State, action: Array) -> core.State:
    state = state.replace(_board=state._board.at[action].set(state._turn))  # type: ignore
    won = _win_check(state._board, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    state = state.replace(
        current_player=(state.current_player + 1) % 2,
        legal_action_mask=state._board < 0,
        rewards=reward,
        terminated=won | jnp.all(state._board != -1),
        _turn=(state._turn + 1) % 2,
    )

    # jax.debug.print('player {} {} player_won {}', action, state._board, won)
    hash_board = hash_state(state)
    ai_action = action_book[hash_board]
    jax.debug.print('ai {} {} {} {}', state.current_player, state._board, hash_board, ai_action, )
    ai_action = jnp.where(~won, ai_action, 0)
    ai_state = state.replace(_board=state._board.at[ai_action].set(state._turn))
    ai_won = _win_check(state._board, state._turn)
    ai_reward = jax.lax.cond(
        ai_won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    ai_state = ai_state.replace(
        current_player=(ai_state.current_player + 1) % 2,
        legal_action_mask=ai_state._board < 0,
        rewards=ai_reward,
        terminated=ai_won | jnp.all(ai_state._board != -1),
        _turn=(ai_state._turn + 1) % 2,
    )
    state = jax.tree.map(
        lambda state, ai_state : jnp.where(won, state, ai_state), state,ai_state
    )
    # jax.debug.print('ai {} {} ai_won {}', ai_action, state._board, ai_won)
    return state

class TicTacToeSinglePlayer(pgx.tic_tac_toe.TicTacToe):

    def _step(self, state: core.State, action: Array, key) -> pgx.tic_tac_toe.State:
        del key
        assert isinstance(state, pgx.tic_tac_toe.State)
        return _step(state, action)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button
    import pgx.tic_tac_toe
    from pgx import core
    from jax import Array
    import jax.numpy as jnp
    from pgx.tic_tac_toe import *

    class TicTacToeGUI:
        def __init__(self):
            self.env = TicTacToeSinglePlayer()
            self.state = self.env.init(jax.random.PRNGKey(0))
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.buttons = []
            self.setup_board()

        def setup_board(self):
            self.ax.clear()
            self.ax.set_xlim(0, 3)
            self.ax.set_ylim(0, 3)
            self.ax.set_aspect('equal')
            self.ax.axis('off')

            for i in range(3):
                for j in range(3):
                    button_ax = plt.axes([j / 3, (2 - i) / 3, 1 / 3, 1 / 3])
                    button = Button(button_ax, '')
                    button.on_clicked(self.make_move(i * 3 + j))
                    self.buttons.append(button)

            self.update_board()

        def make_move(self, action):
            def move(event):
                if not self.state.terminated and self.state._board[action] == -1:
                    self.state = self.env.step(self.state, jnp.array(action))
                    self.update_board()
                    if self.state.terminated:
                        self.show_result()

            return move

        def update_board(self):
            for i, button in enumerate(self.buttons):
                value = self.state._board[i]
                if value == -1:
                    button.label.set_text('')
                elif value == 0:
                    button.label.set_text('X')
                else:
                    button.label.set_text('O')
            plt.draw()

        def show_result(self):
            result = "It's a draw!"
            if np.any(self.state.rewards == 1):
                winner = 'X' if self.state.rewards[0] == 1 else 'O'
                result = f"{winner} wins!"
            self.ax.text(1.5, 1.5, result, ha='center', va='center', fontsize=24)
            plt.draw()

        def play(self):
            plt.show()

    game = TicTacToeGUI()
    game.play()