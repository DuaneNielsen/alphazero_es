import eval.tic_tac_toe_single_player as sp
from jax.random import PRNGKey, split
from pgx.experimental import act_randomly
import jax
import jax.numpy as jnp
from pgx.tic_tac_toe import State
from collections import defaultdict

E, X, O = -1, 0, 1


def verify_action(player, board, expected_action):
    board = jnp.array(board
                      ).reshape(9)
    player = jnp.array(player)
    state = State(_board=board, current_player=player)
    hashcode = sp.hash_state(state)
    action = sp.action_book[hashcode]
    assert action.item() == expected_action, f'expected {expected_action} but got {action} for {board}'


def test_lookups():
    verify_action(
        X,
        [
            [E, E, E],
            [E, E, E],
            [E, E, E]],
        4
    )

    verify_action(
        O,
        [
            [E, E, E],
            [E, E, E],
            [E, E, E]],
        4
    )


def print_board(state):
    print(state._board.reshape(3, 3))

"""
player is player 0, ai is player 1
"""

def test_random_moves():

    results = defaultdict(int)

    for seed in range(100):
        env = sp.TicTacToeSinglePlayer()
        rng = PRNGKey(seed)
        state = env.init(rng)
        random_player = state.current_player.item()
        ai_player = (state.current_player.item() + 1) % 2

        while not state.terminated.all():
            rng, rng_action = split(rng)
            action = act_randomly(rng_action, state.legal_action_mask)
            state = env.step(state, action)

        results['ai_win'] += state.rewards[ai_player].item() == 1
        results['random_win'] += state.rewards[random_player].item() == 1
        results['draw'] += state.rewards[0].item() == 0
        assert state.rewards[ai_player].item() != -1.0, f'seed {seed} caused ai to lose'
    assert sum(results.values()) == 100
    print(results)

def test_random_moves_batched():
    env = sp.TicTacToeSinglePlayer()
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    vmap_act_randomly = jax.vmap(act_randomly)
    batch_size = 10
    rng = PRNGKey(0)
    rng_batch_init = split(rng, batch_size)
    state = init(rng_batch_init)
    ai_player = (state.current_player + 1) % 2

    while not state.terminated.all():
        rng, rng_action = split(rng)
        rng_batch_action = split(rng_action, batch_size)
        action = vmap_act_randomly(rng_batch_action, state.legal_action_mask)
        state = step(state, action)

        jax.debug.print('{}', state.terminated)
        jax.debug.print('{}',state.rewards[jnp.arange(batch_size), ai_player])
