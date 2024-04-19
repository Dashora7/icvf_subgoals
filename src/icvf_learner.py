from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field

import flax
import flax.linen as nn
import ml_collections
from icecream import ic
import functools

state_max = jnp.array([37.727192, 25.744162, 1.362225, 0.99999833, 0.9996134, 0.9998976, 1., 0.6688401, 1.3581934,
                    0.666928, 0.09978515, 0.66525906, 0.09972495, 0.6649802, 1.3628705, 3.97994, 3.8296807,
                    3.2464945, 7.7667384, 6.9804316, 6.992314, 7.5553646, 8.838728, 7.5273356, 6.362007,
                    7.4882784, 6.34013, 7.5405893, 8.736485])
state_min = jnp.array([-1.147686, -1.3210605, 0.19845456, -0.9999111, -0.9992996, -0.9997642, -0.99993134,
                    -0.66625994, -0.09991664, -0.66768396, -1.3384221, -0.6675096, -1.3393451, -0.6663508,
                    -0.09976307, -3.9992015, -4.3275023, -4.2405367, -6.6633897, -6.935104, -6.61271,
                    -7.5409, -6.480048, -7.479568, -8.499193, -7.5485454, -7.049403, -7.5065255, -6.3819485])
# make uniform sampler
random_state_sampler = functools.partial(jax.random.uniform, minval=state_min, maxval=state_max)



def expectile_loss(adv, diff, expectile=0.8):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2

def icvf_loss(value_fn, target_value_fn, batch, config):

    assert all([k in config for k in ['no_intent', 'min_q', 'expectile', 'discount']]), 'Missing ICVF config keys'

    if config['no_intent']:
        batch['desired_goals'] = jax.tree_map(jnp.ones_like, batch['desired_goals'])

    ###
    # Compute TD error for outcome s_+
    # 1(s == s_+) + V(s', s_+, z) - V(s, s_+, z)
    ###

    (next_v1_gz, next_v2_gz) = target_value_fn(batch['next_observations'], batch['goals'], batch['desired_goals'])
    q1_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v1_gz
    q2_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v2_gz
    q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = value_fn(batch['observations'], batch['goals'], batch['desired_goals'])

    ###
    # Compute the advantage of s -> s' under z
    # r(s, z) + V(s', z, z) - V(s, z, z)
    ###

    (next_v1_zz, next_v2_zz) = target_value_fn(batch['next_observations'], batch['desired_goals'], batch['desired_goals'])
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2
    
    q_zz = batch['desired_rewards'] + config['discount'] * batch['desired_masks'] * next_v_zz

    (v1_zz, v2_zz) = target_value_fn(batch['observations'], batch['desired_goals'], batch['desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2
    adv = q_zz - v_zz

    if config['no_intent']:
        adv = jnp.zeros_like(adv)
    
    ###
    #
    # If advantage is positive (next state is better than current state), then place additional weight on
    # the value loss. 
    #
    ##
    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    
    
    ###
    # Push the ICVF to be quasi-metric through regularizing overshoots in triangle inequality
    # if s+ is optimal, we want them to be the same
    # if s+ is suboptimal, we want the sum to be less.
    ###
    
    # TODO: where do we want grad to be flowing...?
    current_v_zz1, current_v_zz2 = value_fn(batch['observations'], batch['desired_goals'], batch['desired_goals'])
    current_v_zz = (current_v_zz1 + current_v_zz2) / 2
    v_splus_z1, v_splus_z2 = value_fn(batch['goals'], batch['desired_goals'], batch['desired_goals'])
    v_splus_z = (v_splus_z1 + v_splus_z2) / 2
    v_s_splus1, v_s_splus2 = value_fn(batch['observations'], batch['goals'], batch['goals'])
    v_s_splus = (v_s_splus1 + v_s_splus2) / 2
    # Compute optimality heuristic
    # current_v_sgz1, current_v_sgz2 = value_fn(batch['observations'], batch['goals'], batch['desired_goals'])
    # current_v_sgz = (current_v_sgz1 + current_v_sgz2) / 2
    QM_C = jax.nn.relu((v_s_splus + v_splus_z) - current_v_zz)
    QM_C -= 0.15 * (QM_C > 0) * current_v_zz # counteract force of making value big negative value
    
    ### 
    # Make the Value function generally conservative
    ###
    # For now, randomly sample a set of states.
    
    # sample n states PER goal....
    
    """num_states = 10
    rng = jax.random.PRNGKey(0)
    random_states = random_state_sampler(shape=(num_states * batch['observations'].shape[0], 29), key=rng)
    obses = jnp.repeat(batch['observations'], num_states, axis=0)
    zs = jnp.repeat(batch['desired_goals'], num_states, axis=0)
    v_sxg1, v_sxg2 = value_fn(obses, random_states, zs)
    v_sxg = ((v_sxg1 + v_sxg2) / 2).reshape(batch['observations'].shape[0], num_states, 1)
    v_sxg_lse = jax.scipy.special.logsumexp(v_sxg, axis=1)
    
    v_sgg1, v_sgg2 = value_fn(batch['observations'], batch['desired_goals'], batch['desired_goals'])
    
    v_sgg = (v_sgg1 + v_sgg2) / 2
    # get logsumexp
    
    C_C =  v_sxg_lse - v_sgg
    C_C = 10*jax.nn.relu(C_C)"""
    
    
    ###
    # Compute loss
    ###
    value_loss = value_loss1 + value_loss2 + QM_C.mean() # + C_C.mean()   # + QM_C.mean()

    def masked_mean(x, mask):
        return (x * mask).sum() / (1e-5 + mask.sum())

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v_gz max': v1_gz.max(),
        'v_gz min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        # 'v_g': v1_g.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
        'reward mean': batch['rewards'].mean(),
        'mask mean': batch['masks'].mean(),
        'q_gz max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
        'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
    }

def periodic_target_update(
    model: TrainState, target_model: TrainState, period: int
) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: optax.periodic_update(p, tp, model.step, period),
        model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)

def vf_loss(agent, batch, value_params):
    ###
    # Compute TD error for outcome g
    # r(s) + gamma * V(s') - V(s)
    ###

    (next_v1_gz, next_v2_gz) = agent.target_value(batch['next_observations'])
    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1_gz
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2_gz

    (v1, v2) = agent.value(batch['observations'], params=value_params)
    
    value_loss1 = ((q1 - v1)**2).mean()
    value_loss2 = ((q2 - v2)**2).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss
    }

class VFAgent(flax.struct.PyTreeNode):
    value: TrainState
    target_value: TrainState
    config: dict = nonpytree_field()
        
    @functools.partial(jax.pmap, axis_name='pmap')
    def update(agent, pretrain_batch):
        def value_loss_fn(value_params):
            return vf_loss(agent, pretrain_batch, value_params)
        
        new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True, pmap_axis='pmap')
        return agent.replace(target_value=new_target_value, value=new_value), value_info
    
    @jax.jit
    def update_single(agent, pretrain_batch):
        def value_loss_fn(value_params):
            return vf_loss(agent, pretrain_batch, value_params)
        
        new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        return agent.replace(target_value=new_target_value, value=new_value), value_info    


    @functools.partial(jax.pmap, axis_name='pmap')
    def get_debug_metrics(agent, batch):
        def get_info(s):
            if agent.config['no_intent']:
                z = jax.tree_map(jnp.ones_like, z)
            return {'v': agent.value(s)[0]}
        s = batch['observations']
        info_s = get_info(s)
        stats = {
            'v_s': info_s['v'].mean()
        }
        stats = jax.lax.pmean(stats, axis_name='pmap')
        return stats

class ICVFAgent(flax.struct.PyTreeNode):
    # rng: jax.random.PRNGKey
    value: TrainState
    target_value: TrainState
    config: dict = nonpytree_field()
        
    @jax.jit
    def update(agent, batch):
        def value_loss_fn(value_params):
            value_fn = lambda s, g, z: agent.value(s, g, z, params=value_params)
            target_value_fn = lambda s, g, z: agent.target_value(s, g, z)

            return icvf_loss(value_fn, target_value_fn, batch, agent.config)
        
        if agent.config['periodic_target_update']:
            new_target_value = periodic_target_update(agent.value, agent.target_value, int(1.0 / agent.config['target_update_rate']))
        else:
            new_target_value = target_update(agent.value, agent.target_value, agent.config['target_update_rate'])
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)

        return agent.replace(value=new_value, target_value=new_target_value), value_info
    
def create_learner(
                 seed: int,
                 observations: jnp.ndarray,
                 value_def: nn.Module,
                 optim_kwargs: dict = {
                    'learning_rate': 0.00005,
                    'eps': 0.0003125
                 },
                 discount: float = 0.95,
                 target_update_rate: float = 0.005,
                 expectile: float = 0.9,
                 no_intent: bool = False,
                 min_q: bool = True,
                 periodic_target_update: bool = False,
                 simple_vf=False,
                **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        if simple_vf:
            value_params =  value_def.init(rng, observations).pop('params')
        else:
            value_params =  value_def.init(rng, observations, observations, observations).pop('params')
        
        value = TrainState.create(value_def, value_params, tx=optax.adam(**optim_kwargs))
        target_value = TrainState.create(value_def, value_params)

        config = flax.core.FrozenDict(dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        ))
        if simple_vf:
            return VFAgent(value=value, target_value=target_value, config=config)
        else:
            return ICVFAgent(value=value, target_value=target_value, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'optim_kwargs': {
            'learning_rate': 0.00005,
            'eps': 0.0003125
        }, # LR for vision here. For FC, use standard 1e-3
        'discount': 0.99,
        'expectile': 0.9,  # The actual tau for expectiles.
        'target_update_rate': 0.005,  # For soft target updates.
        'no_intent': False,
        'min_q': True,
        'periodic_target_update': False,
    })

    return config