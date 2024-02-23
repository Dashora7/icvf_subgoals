import os
import sys

# Some hacky code to add code to PYTHONPATH without installing
start_dir = os.path.abspath(os.path.dirname(__file__)+'../../..') # /home/dibya_ghosh/
print(start_dir)
sys.path.append(f'{start_dir}/icvf_video/')

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

from src.bridge import visualization
from src.bridge.dataset import BridgeDataset
from src.bridge.tasks import (ALIASING_DICT, get_task_id_mapping,
                                       get_tasks)

from src import icvf_learner as learner
from src.gc_dataset import GCSDataset
from jaxrl_m.wandb import setup_wandb, default_wandb_config

from src.vision import encoders
from jaxrl_m.vision import data_augmentations

from ml_collections import config_flags
from icecream import ic
from src.ego4d.dataset import GCSEgo4DDataset
from src.ego4d.data_utils import ALIASING_DICT

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '11tasks', 'Name of experiment')
flags.DEFINE_string('save_dir', f'{start_dir}/experiment_output/', 'Logging dir.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')

flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Create visualization interval.')
flags.DEFINE_integer('save_interval', 25000, 'Save parameter interval.')

flags.DEFINE_integer('batch_size', 64, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_enum('encoder', 'impala', encoders.keys(), 'Encoder name.')
flags.DEFINE_enum('icvf_type', 'monolithic', ['multilinear', 'monolithic'], 'Type of ICVF.')

flags.DEFINE_boolean('augment', True, 'Data augmentation?')

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'ego4d_icvf',
    'group': 'icvf',
    'entity': 'dashora7',
    'name': 'icvf_{encoder}_{icvf_type}_{dataset}',
})

gc_config = GCSEgo4DDataset.get_default_config()
gc_config.update({
    'p_randomgoal': 0.1,
    'p_trajgoal': 0.8,
    'p_currgoal': 0.1,
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', gc_config, lock_config=False)

def augment_images(images, rng):
    image_rngs = jax.random.split(rng, images.shape[0])

    def augment(image, rng):
        return data_augmentations.random_crop(image, rng, padding=16)
    
    return jax.vmap(augment)(images, image_rngs)

@jax.pmap
def add_data_augmentations(rng, batch):
    next_rng, rng_s, rng_ns, rng_g, rng_z = jax.random.split(rng, 5)


    batch['observations'] = augment_images(batch['observations'], rng_s)
    batch['next_observations'] = augment_images(batch['next_observations'], rng_ns)
    batch['goals'] = augment_images(batch['goals'], rng_g)
    batch['desired_goals'] = augment_images(batch['desired_goals'], rng_z)

    return next_rng, batch

@jax.jit
def get_value_metrics(agent, batch):
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

def main(_):

    d = len(jax.devices())
    print(f'Detected {d} devices', jax.devices())
    assert FLAGS.batch_size % d == 0, f'Batch size must be divisible by {d}'

    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)    
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    print('Saving to', FLAGS.save_dir)

    # Load data
    
    # This is the correct way to load data
    # (train_tasks, eval_tasks), (target_train_tasks, target_eval_tasks) = get_tasks(
    #     dataset=FLAGS.dataset, target_dataset='', dataset_directory=os.environ['DATA'])

    # Temporary to get you started: point to your own data directory
    # assert os.path.exists('/nfs/nfs1/users/dibya/door-open/train/out.npy'), 'Please point to your own data directory'
    # train_tasks, eval_tasks = ['/nfs/nfs1/users/dibya/door-open/train/out.npy'], ['/nfs/nfs1/users/dibya/door-open/val/out.npy']
    # target_train_tasks, target_eval_tasks = [], []

    # task_id_mapping = get_task_id_mapping(train_tasks + target_train_tasks, ALIASING_DICT)

    gc_dataset = GCSEgo4DDataset(batch_size=FLAGS.batch_size, **FLAGS.gcdataset)
    gc_val_dataset = GCSEgo4DDataset(batch_size=FLAGS.batch_size, **FLAGS.gcdataset)

    ic(FLAGS.config)

    # Create agent

    example_batch = gc_dataset.sample(1)
    ic(jax.tree_map(lambda arr: arr.shape, example_batch))
    
    encoder_def = encoders[FLAGS.encoder]()
    unreplicated_agent = learner.create_learner(
        seed=FLAGS.seed,
        observations=example_batch['observations'], 
        encoder_def=encoder_def,
        icvf_type=FLAGS.icvf_type,
        **FLAGS.config
    )
    agent = replicate(unreplicated_agent)


    def add_with_prefix(d, d_to_add, prefix):
        d.update({prefix+k: v for k, v in d_to_add.items()})


    augmentation_rng = replicate(jax.random.PRNGKey(0))

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        batch = gc_dataset.sample()
        sharded_batch = shard_batch(batch) # Shards batch across devices
        if FLAGS.augment:
            augmentation_rng, sharded_batch = add_data_augmentations(
                augmentation_rng,
                sharded_batch)
        agent, update_info = agent.update(sharded_batch)

        if i % FLAGS.log_interval == 0:
            debug_metrics = agent.get_debug_metrics(sharded_batch)

            unaugmented_batch = shard_batch(gc_dataset.sample())
            _, train2_update_info = agent.update(unaugmented_batch)
            train2_debug_metrics = agent.get_debug_metrics(unaugmented_batch)

            val_batch = shard_batch(gc_val_dataset.sample())
            _, val_update_info = agent.update(val_batch)
            val_debug_metrics = agent.get_debug_metrics(val_batch)


            train_metrics = dict(iteration=i)
            add_with_prefix(train_metrics, unreplicate(update_info), 'training/')
            add_with_prefix(train_metrics, unreplicate(debug_metrics), 'training/debug/')

            add_with_prefix(train_metrics, unreplicate(train2_update_info), 'training_noaugment/')
            add_with_prefix(train_metrics, unreplicate(train2_debug_metrics), 'training_noaugment/debug/')
            
            add_with_prefix(train_metrics, unreplicate(val_update_info), 'validation/')
            add_with_prefix(train_metrics, unreplicate(val_debug_metrics), 'validation/debug/')

            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            unreplicated_agent = unreplicate(agent)

            what_to_visualize = [
                partial(visualization.visualize_metric, metric_name=metric_name)
                for metric_name in ['rewards', 'v', 'target_q', 'td_error']
            ]
 
            PADDED_SIZE = 100
            train_trajectory, val_trajectory = None, None
            while train_trajectory is None or len(train_trajectory['rewards']) < 16:
                train_trajectory = gc_dataset.get_trajectory()
            while val_trajectory is None or len(val_trajectory['rewards']) < 16:
                val_trajectory = gc_val_dataset.get_trajectory()

            for name, traj in ({
                'train_dataset': train_trajectory,
                'val_dataset': val_trajectory,
            }).items():
                size = len(traj['rewards'])
                def tile(A, reps):
                    return jnp.tile(A[None], (reps, *((1,) * len(A.shape))))
                
                for (goal_name, loc) in [('final', min(size, PADDED_SIZE)-1), ('middle', min(size, PADDED_SIZE) // 2)]:
                    traj['goals'] = jax.tree_map(
                        lambda arr: tile(arr[loc], size),
                        traj['observations']
                    )

                    traj['rewards'] = (jnp.arange(size) == loc).astype(jnp.float32) * FLAGS.gcdataset.reward_scale + FLAGS.gcdataset.reward_shift
                    traj['masks'] = (jnp.arange(size) != loc)

                    padded_batch = to_size(traj, PADDED_SIZE)
                    

                    metrics = get_value_metrics(unreplicated_agent, padded_batch)
                    metrics = {k: v[:size] for k, v in metrics.items()}

                    for k in ['rewards', 'masks']:
                        metrics[k] = traj[k]

                    image = visualization.make_visual(
                        traj['observations'],
                        metrics,
                        what_to_visualize=what_to_visualize
                    )
                    wandb.log({
                        f'visualizations/{name}_{goal_name}' : wandb.Image(image),     
                    }, step=i)

        if i % FLAGS.save_interval == 0:
            unreplicated_agent = unreplicate(agent)
            save_dict = dict(
                agent=flax.serialization.to_state_dict(unreplicated_agent),
                config=FLAGS.config.to_dict()
            )

            fname = os.path.join(FLAGS.save_dir, f'params.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)


def to_size(x, size):
    if isinstance(x, dict):
        return {k: to_size(v, size) for k, v in x.items()}
    if len(x) > size:
        return x[:size]
    else:
        return np.concatenate([
            x, np.zeros((size - len(x), *x.shape[1:]))],
            axis=0
        )

if __name__ == '__main__':
    app.run(main)