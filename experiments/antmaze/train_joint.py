import os
from absl import app, flags
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax

import tqdm
from src import icvf_learner as learner
from src.icvf_networks import icvfs, create_icvf
from icvf_envs.antmaze import d4rl_utils, d4rl_ant, ant_diagnostics, d4rl_pm
from src.gc_dataset import GCSDataset, DiffusionProposalDataset
from src import viz_utils
from src.subgoal_diffuser import GCDDPMBCAgent
from src.icvf_networks import LayerNormMLP

from jaxrl_m.wandb import setup_wandb, default_wandb_config
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate, evaluate_with_trajectories

from ml_collections import config_flags
import pickle
from jaxrl_m.dataset import Dataset
from icecream import ic

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', 'Environment name.')

flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Metric logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Visualization interval.')
flags.DEFINE_integer('save_interval', 100000, 'Save interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer('start_icvf_improve', int(1e6), 'Number of training steps before we use diffusion targets.')
flags.DEFINE_integer('start_diffusion_improve', int(1e5), 'Number of training steps before we use diffusion targets.')


flags.DEFINE_enum('icvf_type', 'multilinear', list(icvfs), 'Which model to use.')
flags.DEFINE_list('hidden_dims', [256, 256], 'Hidden sizes.')


def update_dict(d, additional):
    d.update(additional)
    return d

wandb_config = update_dict(
    default_wandb_config(),
    {
        'project': 'joint_antmaze',
        'group': 'icvf',
        'entity': 'dashora7',
        'name': 'joint_{icvf_type}_{env_name}',
    }
)

config = update_dict(
    learner.get_default_config(),
    {
    'discount': 0.99, 
     'optim_kwargs': { # Standard Adam parameters for non-vision
            'learning_rate': 3e-4,
            'eps': 1e-8
        }
    }
)

icvf_gcdataset_config = GCSDataset.get_default_config()
diffusion_gcdataset_config = GCSDataset.get_default_config()
diffusion_gcdataset_config['p_samegoal'] = 0.0
diffusion_gcdataset_config['p_currgoal'] = 0.0
diffusion_gcdataset_config['p_randomgoal'] = 1.0
diffusion_gcdataset_config['p_trajgoal'] = 0.0
diffusion_gcdataset_config['intent_sametraj'] = False

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config, lock_config=False)
config_flags.DEFINE_config_dict('icvf_gcdataset', icvf_gcdataset_config, lock_config=False)
config_flags.DEFINE_config_dict('diffusion_gcdataset', diffusion_gcdataset_config, lock_config=False)

def main(_):
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    dataset = d4rl_utils.get_dataset(env)
    icvf_gc_dataset = GCSDataset(dataset,
        **FLAGS.icvf_gcdataset.to_dict())
    diffusion_gc_dataset = GCSDataset(dataset,
        **FLAGS.diffusion_gcdataset.to_dict(), hiql_mode=True)
    
    
    ### ICVF INIT ###
    example_batch = icvf_gc_dataset.sample(1)
    hidden_dims = tuple([int(h) for h in FLAGS.hidden_dims])
    value_def = create_icvf(FLAGS.icvf_type, hidden_dims=hidden_dims)
    icvf_agent = learner.create_learner(FLAGS.seed,
                    example_batch['observations'],
                    value_def,
                    **FLAGS.config)
    icvf_fn = jax.jit(lambda a, b, c: icvf_agent.value(a, b, c).sum(0))
    
    ### DIFFUSION INIT ###
    example_batch = diffusion_gc_dataset.sample(1)
    hidden_dims = tuple([int(h) for h in FLAGS.hidden_dims])
    encoder_def = LayerNormMLP(hidden_dims)
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    diff_agent = GCDDPMBCAgent.create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"], 
        encoder_def=encoder_def,
        conditional=True
    )
    
    ## DEFINE GENERATIVE DATASET ##
    generation_gc_dataset = DiffusionProposalDataset(
        diffusion_agent=diff_agent,
        one_shot=True,
        icvf_agent=icvf_agent,
        eps=5)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        # TODO: finalize heuristic/weighting function for diffusion train.
        # this is important, it specified the distribution of subgoals we want
        # to improve the ICVF on. Possible just weight exponentially by V(s, s+, g)
        
        # ICVF DATA LEARN
        batch = icvf_gc_dataset.sample(FLAGS.batch_size)  
        icvf_agent, icvf_update_info = icvf_agent.update(batch)
        
        # DIFFUSION IMPROVE
        if i > FLAGS.start_diffusion_improve:
            batch = diffusion_gc_dataset.sample(FLAGS.batch_size)
            diff_agent, diff_update_info = diff_agent.update(batch, icvf_fn=icvf_fn)
        
        # ICVF IMPROVE
        if i > FLAGS.start_icvf_improve:
            batch = generation_gc_dataset.generate(FLAGS.batch_size)
            icvf_agent, update_info = icvf_agent.update(batch)
        
        
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in icvf_update_info.items()}
            if i > FLAGS.start_diffusion_improve:
                train_metrics.update({f'training/diffusion_{k}': v for k, v in diff_update_info.items()})
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            metrics = []
            rng, val_key = jax.random.split(rng)
            metrics.append(icvf_agent.get_debug_metrics(batch, seed=val_key))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb.log({"validation_icvf": metrics}, step=i)
            
            metrics = []
            rng, val_key = jax.random.split(rng)
            metrics.append(diff_agent.get_debug_metrics(batch, seed=val_key))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb.log({"validation_diffusion": metrics}, step=i)

        if i % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(icvf_agent),
                config=FLAGS.config.to_dict()
            )
            fname = os.path.join(FLAGS.save_dir, f'icvf_params.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
            
            save_dict = dict(
                agent=flax.serialization.to_state_dict(diff_agent),
                config=FLAGS.config.to_dict()
            )
            fname = os.path.join(FLAGS.save_dir, f'diffusion_params.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)



if __name__ == '__main__':
    app.run(main)

