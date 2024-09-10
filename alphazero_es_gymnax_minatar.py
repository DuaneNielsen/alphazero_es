import evosax.core
import gymnax
import flax.linen as nn
from argparse import ArgumentParser
import functools
from typing import Any, Callable, Sequence, Tuple, Optional, Sequence, Dict, Union, Iterable
import mctx
import warnings
from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
from evosax import ParameterReshaper, FitnessShaper, OpenES

warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle

import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

parser = ArgumentParser()

parser.add_argument('--aznet_channels', type=int, default=8)
parser.add_argument('--aznet_blocks', type=int, default=5)
parser.add_argument('--aznet_layernorm', type=str, default='None')
# parser.add_argument('--num_hidden_units', type=int, default=16)
# parser.add_argument('--num_linear_layers', type=int, default=1)
parser.add_argument('--num_output_units', type=int, default=3)
parser.add_argument('--output_activation', type=str, default='categorical')
parser.add_argument('--env_name', type=str, default='Breakout-MinAtar')
parser.add_argument('--popsize', type=int, default=1600)
parser.add_argument('--sigma_init', type=float, default=0.03)
parser.add_argument('--sigma_decay', type=float, default=1.0)
parser.add_argument('--sigma_limit', type=float, default=0.01)
parser.add_argument('--lrate_init', type=float, default=0.05)
parser.add_argument('--lrate_decay', type=float, default=1.0)
parser.add_argument('--lrate_limit', type=float, default=0.001)
parser.add_argument('--num_generations', type=int, default=320)
parser.add_argument('--num_mc_evals', type=int, default=1)
parser.add_argument('--network', type=str, default='AZnet')  # so it appears as a hyperparameter in wandb
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_simulations', type=int, default=16)
parser.add_argument('--visualize_off', action='store_true')
parser.add_argument('--max_epi_len', type=int, default=1000)
parser.add_argument('--opt_name', type=str, choices=evosax.core.GradientOptimizer.keys(), default='sgd')
parser.add_argument('--boosted_eval_num_simulations', type=int, default=None, help='evaluate best scores a second time with this many simulations')
args = parser.parse_args()

rng = jax.random.PRNGKey(args.seed)


class ResnetV2Block(nn.Module):
    features: int
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    block_name: str = None
    dtype: str = 'float32'
    param_dict: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        i = x

        if args.aznet_layernorm == 'layernorm':
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding=((1, 1), (1, 1)),
            kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                self.param_dict['conv1']['weight']),
            use_bias=False,
            dtype=self.dtype
        )(x)

        if args.aznet_layernorm == 'layernorm':
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding=((1, 1), (1, 1)),
            kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                self.param_dict['conv2']['weight']),
            use_bias=False,
            dtype=self.dtype
        )(x)
        return x + i


class AZNet(nn.Module):
    num_actions: int = args.num_output_units
    num_channels: int = args.aznet_channels
    num_blocks: int = args.aznet_blocks
    dtype: str = 'float32'
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    param_dict: Optional[Dict] = None

    @nn.compact
    def __call__(self, x, rng):
        x = x.astype(self.dtype)
        x = nn.Conv(
            features=self.num_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=self.kernel_init if self.param_dict is None
            else lambda *_: jnp.array(self.param_dict['conv1']['weight']),
            use_bias=False,
            dtype=self.dtype,
        )(x)

        if args.aznet_layernorm == 'layernorm':
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.relu(x)

        for block in range(self.num_blocks):
            x = ResnetV2Block(
                features=self.num_channels,
            )(x)

        if args.aznet_layernorm == 'layernorm':
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.relu(x)

        # policy and value heads here, below code is placeholder for ES testing
        x = x.reshape(-1)
        policy = nn.Dense(features=self.num_actions,
                          kernel_init=self.kernel_init if self.param_dict is None
                          else lambda *_: jnp.array(self.param_dict['fc1']['weight']),
                          bias_init=self.bias_init if self.param_dict is None
                          else lambda *_: jnp.array(self.param_dict['fc1']['bias']),
                          dtype=self.dtype,
                          )(x)

        value = nn.Dense(features=1,
                         kernel_init=self.kernel_init if self.param_dict is None
                         else lambda *_: jnp.array(self.param_dict['fc2']['weight']),
                         bias_init=self.bias_init if self.param_dict is None
                         else lambda *_: jnp.array(self.param_dict['fc2']['bias']),
                         dtype=self.dtype,
                         )(x)

        return policy, value

        # x_out = jax.random.categorical(rng, x)
        # return x_out


# Create placeholder params for env
env, env_params = gymnax.make(args.env_name)
pholder = jnp.zeros(env.observation_space(env_params).shape)
model = AZNet(
    num_actions=args.num_output_units
)
init_policy_params = model.init(
    rng,
    x=pholder,
    rng=rng
)

"""Rollout wrapper for gymnax environments."""

import functools
from typing import Any, Optional
import jax
import jax.numpy as jnp
import gymnax


class RolloutWrapper(object):
    """Wrapper to define batch evaluation for generation parameters."""

    def __init__(
            self,
            model_forward=None,
            env_name: str = "Pendulum-v1",
            num_env_steps: Optional[int] = None,
            env_kwargs: Any | None = None,
            env_params: Any | None = None,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env_name = env_name
        # Define the RL environment & network forward function
        if env_kwargs is None:
            env_kwargs = {}
        if env_params is None:
            env_params = {}
        self.env, self.env_params = gymnax.make(self.env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.model_forward = model_forward

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    @functools.partial(jax.jit, static_argnums=(0, 2,))
    def population_rollout(self, rng_eval, num_simulations, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, None, 0))
        return pop_rollout(rng_eval, num_simulations, policy_params)

    @functools.partial(jax.jit, static_argnums=(0, 2,))
    def batch_rollout(self, rng_eval, num_simulations, policy_params):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None, None))
        return batch_rollout(rng_eval, num_simulations, policy_params)

    @functools.partial(jax.jit, static_argnums=(0, 2,))
    def single_rollout(self, rng_input, num_simulations, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def recurrent_fn(model_params, rng_key: jnp.ndarray, action: jnp.ndarray, state_input):
            # model: params
            # state: embedding

            rng_key, rng_step, rng_net = jax.random.split(rng_key, 3)
            obs, state, policy_params, rng, cum_reward, valid_mask, step, done = (
                jax.tree.map(lambda s: jnp.squeeze(s, axis=0), state_input))
            # model_params, model_state = model
            # current_player = obs.current_player

            # step the environment
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action.squeeze(0), self.env_params
            )

            # store the transition and reward
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            state_input = [next_obs, next_state, policy_params, rng_key, new_cum_reward, new_valid_mask, step + 1, done]
            state_input = jax.tree.map(lambda s: jnp.expand_dims(s, axis=0), state_input)

            # compute the logits and values for the next state
            logits, value = self.model_forward(model_params, obs, rng_net)

            # mask invalid actions
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            # logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

            # reward = state.rewards  # [jnp.arange(state.rewards.shape[0]), current_player]
            value = jnp.where(done, 0.0, value)
            # discount = -1.0 * jnp.ones_like(value)
            discount = 0.99 * jnp.ones_like(value)
            discount = jnp.where(done, 0.0, discount)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=jnp.expand_dims(reward, axis=0),
                discount=discount,
                prior_logits=jnp.expand_dims(logits, axis=0),
                value=value,
            )
            return recurrent_fn_output, state_input

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask, step, done = (
                jax.tree.map(lambda s: jnp.squeeze(s, axis=0), state_input))

            key1, rng_step, net_rng, rng_action = jax.random.split(rng, 4)

            logits, value = self.model_forward(policy_params, obs, net_rng)

            logits = jnp.expand_dims(logits, axis=0)

            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state_input)

            policy_output = mctx.gumbel_muzero_policy(
                params=policy_params,
                rng_key=key1,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=jnp.zeros_like(logits),
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            # action = jax.random.categorical(rng_action, jnp.log(policy_output.action_weights), axis=-1)
            action = policy_output.action.squeeze(0)
            # keys = jax.random.split(key2, batch_size)
            # step the environment
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)

            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
                step + 1,
                done,
            ]
            carry = jax.tree.map(lambda s: jnp.expand_dims(s, axis=0), carry)
            y = [obs, state, action, reward, next_obs, done]
            return carry, y

        def early_termination_loop_with_trajectory(policy_step, max_steps, initial_state):
            def cond_fn(carry):
                state, _ = carry
                _, _, _, _, _, _, step, done = state
                return jnp.logical_and(jnp.logical_not(jnp.all(done)), jnp.any(step < max_steps))

            def body_fn(carry):
                state, trajectory = carry
                next_state, step_output = policy_step(state, ())
                step_counter = next_state[6]

                updated_trajectory = jax.tree_util.tree_map(
                    lambda t, s: t.at[step_counter - 1].set(s),
                    trajectory,
                    step_output
                )

                return (next_state, updated_trajectory)

            trajectory_template = policy_step(initial_state, ())[1]
            initial_trajectory = jax.tree_util.tree_map(
                lambda x: jnp.zeros((max_steps,) + x.shape, dtype=x.dtype),
                trajectory_template
            )

            initial_carry = (initial_state, initial_trajectory)
            final_state, final_trajectory = jax.lax.while_loop(cond_fn, body_fn, initial_carry)

            return final_state, final_trajectory

        state_input = [
            obs,
            state,
            policy_params,
            rng_episode,
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array(0),
            jnp.array(False),
        ]
        state_input = jax.tree.map(lambda s: jnp.expand_dims(s, axis=0), state_input)

        # Run the early termination loop
        carry_out, scan_out = early_termination_loop_with_trajectory(policy_step, args.max_epi_len, state_input)

        # Return the sum of rewards accumulated by agent in episode rollout
        obs, state, action, reward, next_obs, done = scan_out
        cum_return, steps = carry_out[-4], carry_out[-2]
        return obs, state, action, reward, next_obs, done, cum_return, steps

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, _ = self.env.reset(rng, self.env_params)
        return obs.shape


# Define rollout manager for env
manager = RolloutWrapper(model.apply, env_name=args.env_name, num_env_steps=args.max_epi_len)

# from evosax import OpenES

# Helper for parameter reshaping into appropriate datastructures
param_reshaper = ParameterReshaper(init_policy_params, n_devices=1)

# Instantiate and initialize the evolution strategy
strategy = OpenES(popsize=args.popsize,
                  num_dims=param_reshaper.total_params,
                  opt_name=args.opt_name)

es_params = strategy.default_params
es_params = es_params.replace(sigma_init=args.sigma_init, sigma_decay=args.sigma_decay, sigma_limit=args.sigma_limit)
es_params = es_params.replace(opt_params=es_params.opt_params.replace(
    lrate_init=args.lrate_init, lrate_decay=args.lrate_decay, lrate_limit=args.lrate_limit))

es_state = strategy.initialize(rng, es_params)

fit_shaper = FitnessShaper(maximize=True, centered_rank=True)

import wandb

wandb.init(project=f'alphazero-es-{args.env_name}', config=args.__dict__, settings=wandb.Settings(code_dir="."))

# num_generations = 100
# num_mc_evals = 128
print_every_k_gens = 1
best_eval = -jnp.inf
total_frames = 0

for gen in range(args.num_generations):
    rng, rng_init, rng_ask, rng_rollout, rng_eval = jax.random.split(rng, 5)
    # Ask for candidates to evaluate
    x, es_state = strategy.ask(rng_ask, es_state)

    # Reshape parameters into flax FrozenDicts
    reshaped_params = param_reshaper.reshape(x)
    rng_batch_rollout = jax.random.split(rng_rollout, args.num_mc_evals)

    # Perform population evaluation
    _, _, _, _, _, _, cum_ret, steps = manager.population_rollout(rng_batch_rollout, args.num_simulations,
                                                                  reshaped_params)

    # Mean over MC rollouts, shape fitness and update strategy
    fitness = cum_ret.mean(axis=1).squeeze()
    fit_re = fit_shaper.apply(x, fitness)
    es_state = strategy.tell(x, fit_re, es_state)

    # param_new = param_prev - grads * state.lrate
    total_frames += steps.sum().item()

    log = {'score_hist': fitness, 'score': fitness.mean(), 'max_steps': jnp.max(steps), 'steps_hist': steps,
           'total_frames': total_frames}
    status = f"Generation: {gen + 1} fitness: {fitness.mean():.3f}, max_steps: {jnp.max(steps)}, total frames: {total_frames}"

    if (gen + 1) % print_every_k_gens == 0:

        eval_params = param_reshaper.reshape(jnp.expand_dims(es_state.mean, axis=0))
        eval_params = jax.tree.map(lambda p: p.squeeze(0), eval_params)

        def visualize(state, longest_run, max_steps):
            # visualization
            states = []
            for i in range(max_steps.item()):
                states.append(jax.tree.map(lambda x: x[longest_run, i], state))

            from gymnax.visualize import Visualizer
            vis = Visualizer(env, env_params, states)
            anim_path = f"{wandb.run.dir}/anim.gif"
            vis.animate(anim_path)
            log.update({"viz": wandb.Video(anim_path, fps=8, format='gif')})

        def evaluate(rng_rollout, num_evals, num_simulations):

            rng_batch_eval_rollout = jax.random.split(rng_rollout, num_evals)
            obs, state, action, reward, next_obs, done, cum_ret, steps = \
                manager.batch_rollout(rng_batch_eval_rollout, num_simulations, eval_params)

            eval_score = cum_ret.mean()
            longest_run_steps = jnp.max(steps)
            longest_run_index = jnp.argmax(steps)

            return eval_score, state, longest_run_index, longest_run_steps

        eval_score, state, longest_run_index, longest_run_steps  = \
            evaluate(rng_rollout, 20, args.num_simulations)

        log.update({
            f"eval_score": eval_score,
            f"eval_steps": longest_run_steps,
        })
        status += f' Eval fitness: {eval_score:.3f}, max_steps: {longest_run_steps}'

        # save if the model is better, update the visualization, and try a 1000 step run
        if eval_score.item() > best_eval:
            best_eval = max(best_eval, eval_score.item())
            with open(f'{wandb.run.dir}/best_{best_eval:.2f}.param', 'wb') as f:
                pickle.dump(eval_params, f)

            if not args.visualize_off:
                print('generating visualization')
                visualize(state, longest_run_index, longest_run_steps)

            if args.boosted_eval_num_simulations:

                print(f'evaluating {args.boosted_eval_num_simulations} sim inference')
                eval_score, longest_run_state, longest_run_index, longest_run_steps  = \
                    evaluate(rng_rollout, 20, args.boosted_eval_num_simulations)

                log.update({
                    f"eval_score_{args.boosted_eval_num_simulations}": eval_score,
                    f"eval_steps_{args.boosted_eval_num_simulations}": longest_run_steps,
                })
                status += f', {args.boosted_eval_num_simulations} sim Eval: {eval_score:0.3f} steps: {longest_run_steps}'

        log.update({"best_eval": best_eval})

    wandb.log(log, step=gen)
    print(status)
