import copy
from functools import partial
from typing import Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.serialization import from_state_dict

from jaxrl_m.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.networks import GCEncodingWrapper, EncodingWrapper
from jaxrl_m.typing import Batch, PRNGKey
from jaxrl_m.networks import (
    FourierFeatures,
    ScoreActor,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from jaxrl_m.networks import MLP, MLPResNet
from src import icvf_learner as learner
from src.icvf_networks import icvfs, create_icvf
import pickle

def heuristics_to_weights(advs, vs):
    # step function threshold
    a_threshold = jnp.where(advs > 10, 1, 0)
    v_threshold = jnp.where(vs > 40, 1, 0)
    return a_threshold * v_threshold

def icvf_weights(batch, usefulness=1, reachability=1, icvf_fn=None):
    obs = batch["observations"]
    subgoal = batch["actions"]
    goal = batch["goals"]
    advantages = icvf_fn(subgoal, goal, goal) - icvf_fn(obs, goal, goal)
    values = icvf_fn(obs, subgoal, subgoal) - icvf_fn(obs, goal, goal)
    return heuristics_to_weights(advantages, values)

def ddpm_bc_loss(noise_prediction, noise, weights=None):
    ddpm_loss = jnp.square(noise_prediction - noise).sum(-1)
    if weights is not None:
        ddpm_loss = (ddpm_loss * weights) + 1e-8

    return ddpm_loss.mean(), {
        "ddpm_loss": ddpm_loss,
        "ddpm_loss_mean": ddpm_loss.mean(),
    }

class GCDDPMBCAgent(flax.struct.PyTreeNode):
    """
    Models subgoal distribution with a diffusion model.

    Assumes observation history and goal as input, and subgoal as output.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None, icvf_fn: object = None):
        
        def actor_loss_fn(params, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            noise_pred = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (batch["observations"], batch["goals"]),
                noisy_actions,
                time,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            if icvf_fn is not None:
                return ddpm_bc_loss(
                    noise_pred,
                    noise_sample,
                    weights=icvf_weights(batch, usefulness=1, reachability=1, icvf_fn=icvf_fn) # None
                )
            else:
                return ddpm_bc_loss(
                    noise_pred,
                    noise_sample,
                )
                

        loss_fns = {
            "actor": actor_loss_fn,
        }
        
        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # log learning rates
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info
    
    @jax.jit
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey = None,
        temperature: float = 1.0,
        clip_sampler: bool = True,
    ) -> jnp.ndarray:
        # assert len(observations["image"].shape) > 3, "Must use observation histories"

        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            eps_pred = self.state.apply_fn(
                {"params": self.state.target_params},
                (observations, goals),
                current_x,
                input_time,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hats"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(
                key,
                shape=current_x.shape,
            )
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if clip_sampler:
                
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()

        key, rng = jax.random.split(seed)

        if len(observations.shape) == 1:
            # unbatched input from evaluation
            batch_size = 1
            observations = jax.tree_map(lambda x: x[None], observations)
            goals = jax.tree_map(lambda x: x[None], goals)
        else:
            batch_size = observations.shape[0]

        input_tuple, () = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple

        if batch_size == 1:
            # this is an evaluation call so unbatch
            return action_0[0]
        else:
            return action_0

    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(
            observations=batch["observations"], goals=batch["goals"], seed=seed
        )

        metrics = {
            "mse": ((actions - batch["actions"]) ** 2).sum((-1)).mean(),
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # example arrays for model init
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # agent config
        encoder_def: nn.Module,
        # should only be set if not language conditioned
        shared_goal_encoder: Optional[bool] = True,
        early_goal_concat: Optional[bool] = True,
        # other shared network config
        use_proprio: bool = False,
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
            "use_layer_norm": True
        },
        # optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # DDPM algorithm train + inference config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
        conditional=True
    ):
        # assert len(actions.shape) > 1, "Must use chunking" # WHAT IS THIS?
        if shared_goal_encoder is None or early_goal_concat is None:
            raise ValueError(
                "Shared_goal_encoder and early_goal_concat must be set"
            )

        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        if conditional:
            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            encoder_def = EncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        networks = {
            "actor": ScoreActor(
                encoder_def,
                FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                MLP(
                    (
                        2 * score_network_kwargs["time_dim"],
                        score_network_kwargs["time_dim"],
                    )
                ),
                MLPResNet(
                    score_network_kwargs["num_blocks"],
                    actions.shape[-1], # actions.shape[-2] * actions.shape[-1],
                    dropout_rate=score_network_kwargs["dropout_rate"],
                    use_layer_norm=score_network_kwargs["use_layer_norm"],
                ),
            ),
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        if len(actions.shape) == 2:
            example_time = jnp.zeros((actions.shape[0], 1))
        else:
            example_time = jnp.zeros((1,))
        params = jax.jit(model_def.init)(
            init_rng, actor=[(observations, goals), actions, example_time]
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        config = flax.core.FrozenDict(
            dict(
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
                action_dim=(actions.shape[-1],),
                action_max=np.array([
                    37.727192, 25.744162, 1.362225, 0.99999833, 0.9996134, 0.9998976, 1., 0.6688401, 1.3581934,
                    0.666928, 0.09978515, 0.66525906, 0.09972495, 0.6649802, 1.3628705, 3.97994, 3.8296807,
                    3.2464945, 7.7667384, 6.9804316, 6.992314, 7.5553646, 8.838728, 7.5273356, 6.362007,
                    7.4882784, 6.34013, 7.5405893, 8.736485]),
                action_min=np.array([
                    -1.147686, -1.3210605, 0.19845456, -0.9999111, -0.9992996, -0.9997642, -0.99993134,
                    -0.66625994, -0.09991664, -0.66768396, -1.3384221, -0.6675096, -1.3393451, -0.6663508,
                    -0.09976307, -3.9992015, -4.3275023, -4.2405367, -6.6633897, -6.935104, -6.61271,
                    -7.5409, -6.480048, -7.479568, -8.499193, -7.5485454, -7.049403, -7.5065255, -6.3819485]),
                betas=betas,
                alphas=alphas,
                alpha_hats=alpha_hat,
                diffusion_steps=diffusion_steps,
                action_samples=action_samples,
                repeat_last_step=repeat_last_step,
            )
        )
        return cls(state, config, lr_schedules)