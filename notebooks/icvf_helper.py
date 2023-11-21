import os
import sys
from flax.training import checkpoints
import time
import copy
# Some hacky code to add code to PYTHONPATH without installing
"""start_dir = os.path.abspath('/nfs/nfs1/users/riadoshi') # /home/dibya_ghosh/
print(start_dir)
sys.path.append(f'{start_dir}/icvf_video/')"""

import numpy as np
import jax
import jax.numpy as jnp
import pickle
from functools import partial
from absl import app, flags
import flax

# Utilities for pmap
from jaxrl_m.common import shard_batch
from flax.jax_utils import replicate, unreplicate

import tqdm
import wandb

from src import icvf_learner as learner
from src.gc_dataset import GCSDataset
from jaxrl_m.wandb import setup_wandb, default_wandb_config

from src.vision import encoders
from jaxrl_m.vision import data_augmentations

from src.bridge import visualization
from src.bridge.dataset import BridgeDataset
from src.bridge.tasks import (ALIASING_DICT, get_task_id_mapping,
                                       get_tasks)

from ml_collections import config_flags
from src.helpers import *
from icecream import ic


@jax.jit
def get_value_metrics(agent, batch):
    # import pdb; pdb.set_trace()
    v1, v2 = agent.value(batch['observations'], batch['goals'], batch['goals'])
    nv1, nv2 = agent.value(batch['next_observations'], batch['goals'], batch['goals'])

    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * nv1

    return {
        'v': (v1 + v2) / 2,
        'v1': v1,
        'v2': v2,
        'target_q': q,
        'td_error': (q - v1) ** 2,
    }




def prep_learner(pretrained_encoder_path, encoder_type='ptr_resnet_34_v1', dataset='small_five', num_domains=5, target_dataset='new_cc', drop_task_id=False):
    gc_config = GCSDataset.get_default_config()
    gc_config.update({
        'reward_shift': -1.0,
        'terminal': True,
    })

    config = learner.get_default_config()
    bridge_config = BridgeDataset.get_default_config()

    if dataset == "small_five":
        train_tasks_per_domain, action_spaces = get_small_five()

    if target_dataset == "new_cc":
        target_train_tasks, target_eval_tasks = get_new_toykitchen6_cucumber_in_orange_pot()

    task_id_mapping = multi_get_task_id_mapping(train_tasks_per_domain, target_train_tasks, target_domain=0)

    dataset = BridgeDataset.create(train_tasks_per_domain, task_id_mapping, ALIASING_DICT, bridge_config, num_domains=num_domains, action_spaces = action_spaces, drop_task_id=drop_task_id)
    gc_dataset = GCSDataset(dataset, **gc_config)
    example_batch = gc_dataset.sample(1)
    encoder_def = encoders[encoder_type]()

    # import pdb; pdb.set_trace() # check if exmaple_batch observations has a task id or not, bc the 
    # batch we're feeding into get_value_metric does not have task id

    # if this works (and example batch observations HAS a task id), then we can initialize our 
    # agent the same way as here (bc task id is really irrelevant), and then drop the task id in build batch


    unreplicated_agent = learner.create_learner(
        seed=np.random.choice(1000000),
        observations=example_batch['observations'], 
        encoder_def=encoder_def,
        icvf_type='multilinear',
        use_pretrained_weights=True,
        pretrained_encoder_path=pretrained_encoder_path,
        **config
    )
    # agent = replicate(unreplicated_agent)

    return unreplicated_agent

def build_batch(target_traj, goal="final", custom_goal_traj_idxs=[]):
    trajlen = len(target_traj['observations'])

    image_key='images0'
    if 'image' in target_traj['observations'][0].keys():
        image_key = 'image'

    dones = np.zeros(trajlen)
    dones[-1] = 1

    rewards = np.zeros(trajlen)
    rewards[-1] = 1

    observations = np.array(target_traj['observations'])
    next_observations = np.array(target_traj['next_observations'])

    obses, next_obses = {}, {}
    for obs, next_obs in zip(observations, next_observations):
        if 'image' in obses:
            obses['image'].append(obs[image_key])
            next_obses['image'].append(next_obs[image_key])
        
        else:
            obses['image'] = [obs[image_key]]
            next_obses['image'] = [next_obs[image_key]]
    
    obses['image'] =np.array(obses['image'])
    next_obses['image'] =np.array(next_obses['image'])

    masks = np.ones(trajlen)
    masks[-1] = 0

    if goal=="final":
        goal_image = target_traj['observations'][-1][image_key]
        goals = np.array([goal_image]*trajlen)
    elif goal=="middle":
        goal_image = target_traj['observations'][trajlen//2][image_key]
        goals = np.array([goal_image]*trajlen)
    elif goal == "start":
        goal_image = target_traj['observations'][0][image_key]
        goals = np.array([goal_image]*trajlen)
    elif goal == "custom":
        goals = np.array([obses['image'][idx] for idx in custom_goal_traj_idxs])

    batch = {
        'observations':obses,
        'next_observations':next_obses,
        'rewards':rewards,
        'masks':masks, 
        'goals': {'image': goals},
        'dones_float': dones,
    }

    return batch

def get_small_five():
    train_tasks_bridge, eval_tasks_bridge = get_0515_toykitchen6_sweetpotato_on_plate()
    train_tasks_wx200, eval_tasks_wx200 = get_one_wx200()
    train_tasks_sawyer, eval_tasks_sawyer = get_one_sawyer()
    train_tasks_franka, eval_tasks_franka = get_one_franka()
    train_tasks_yumi, eval_tasks_yumi = get_one_yumi()

    train_tasks_per_domain = [train_tasks_bridge, train_tasks_wx200, train_tasks_sawyer, train_tasks_franka, train_tasks_yumi]
    eval_tasks_per_domain = [eval_tasks_bridge, eval_tasks_wx200, eval_tasks_sawyer, eval_tasks_franka, eval_tasks_yumi]

    action_spaces = [action_space('bridge'), action_space('wx200'), action_space('sawyer2'), action_space('franka'), action_space('yumi')]

    return train_tasks_per_domain, action_spaces