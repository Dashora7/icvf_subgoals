import numpy as np
import tqdm
import collections
from jaxrl_m.dataset import Dataset

def _reshape_image(obs):
    if len(obs.shape) == 1:
        obs = np.reshape(obs, (3, 128, 128))
        return np.transpose(obs, (1, 2, 0))
    elif len(obs.shape) == 3:
        return obs
    else:
        raise ValueError

def load_buffer(dataset_file, num_trajs=None):
    print('loading buffer data from ', dataset_file)    
    try:
        trajs = np.load(f'{dataset_file}', allow_pickle=True)
        if num_trajs is not None:
            np.random.shuffle(trajs)
            trajs = trajs[:num_trajs]
        n_transitions = sum([len(traj['observations']) for traj in trajs])
        print(f'Loading in {n_transitions} transitions')
    except:
        print(f'Failed to load buffer data from {dataset_file}')
        trajs = []
    return trajs

class BridgeDataset(Dataset):
    @staticmethod
    def get_default_config():
        import ml_collections
        config = ml_collections.ConfigDict()
        config.frame_stack = 1
        config.action_queuesize = 1
        config.num_final_reward_steps = 3
        config.add_prev_actions = False
        config.rescale_actions = True
        config.consolidate_state = True
        config.multi_viewpoint = False
        config.all_views_together = False
        config.reward_scale = 11
        config.reward_shift = -1
        config.sim=False
        config.num_trajs = ml_collections.config_dict.placeholder(int)
        return config

    @classmethod
    def create(cls,tasks, task_id_mapping, task_aliasing_dict, config, orig_low=None, orig_high=None):

        all_trajs = []
        for dataset_file in tasks:
            trajs = load_buffer(dataset_file, config.num_trajs)
            task_name = str.split(dataset_file, '/')[-3]
            if task_name in task_aliasing_dict:
                task_name = task_aliasing_dict[task_name]
            for traj in trajs:
                traj['task_description'] = task_name
            all_trajs.extend(trajs)

        print('*'*30)
        n_transitions = sum([len(traj['observations']) for traj in all_trajs])
        print(f'Loading in {n_transitions} transitions')
        print('*' * 30)

        transitions = []
        print('formatting data...')
        for traj in tqdm.tqdm(all_trajs):
            if config.all_views_together:
                image_keys = [f'images{viewpoint}' for viewpoint in range(3)]
                transitions.extend(get_traj_transitions(traj, task_id_mapping, config, image_keys=image_keys, orig_low=orig_low, orig_high=orig_high))
            elif config.sim:
                transitions.extend(get_traj_transitions(traj, task_id_mapping, config, image_keys=[f'image'], orig_low=orig_low, orig_high=orig_high))
            else:
                for viewpoint in range(1 if not config.multi_viewpoint else 3):
                    transitions.extend(get_traj_transitions(traj, task_id_mapping, config, image_keys=[f'images{viewpoint}'], orig_low=orig_low, orig_high=orig_high))
    
        print('Loading into array format')
        transitions_np = dict()
        for k in transitions[0]:
            if 'observations' in k:
                transitions_np[k] = {
                    k2: np.stack([transition[k][k2] for transition in transitions]) for k2 in transitions[0][k] }
            else:
                transitions_np[k] = np.stack([transition[k] for transition in transitions])
        
        if config.consolidate_state:
            state_keys = [k for k in transitions_np['observations'] if 'image' not in k]
            print(f'Consolidating {state_keys} -> "state" ')
                    
            transitions_np['observations']['state'] = np.concatenate([
                transitions_np['observations'].pop(k) for k in state_keys], axis=-1)
            transitions_np['next_observations']['state'] = np.concatenate([
            transitions_np['next_observations'].pop(k) for k in state_keys], axis=-1)

        return cls(transitions_np)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_starts = np.array([0] + list(self._dict['dones_float'].nonzero()[0] + 1))

    def get_trajectory(self, trajectory_indx=None, seed=None):
        if trajectory_indx is None:
            trajectory_indx = np.random.default_rng(seed).choice(len(self.trajectory_starts)- 1)
        indx = np.arange(
            self.trajectory_starts[trajectory_indx],
            self.trajectory_starts[trajectory_indx+1]
        )     
        return self.sample(len(indx), indx=indx)

def is_positive_sample(traj, i, task_name, num_final_reward_steps):
    return i >= len(traj['observations']) - num_final_reward_steps

def get_traj_transitions(traj, task_id_mapping, config, image_keys=['images0'], orig_low=None, orig_high=None):
    frame_stack, action_queuesize, num_final_reward_steps, add_prev_actions, rescale_actions = \
        config.frame_stack, config.action_queuesize, config.num_final_reward_steps, config.add_prev_actions, config.rescale_actions

    if rescale_actions:
        if orig_low is None:
            orig_low = np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.])
        if orig_high is None:
            orig_high = np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0])
        safety_margin = 0.01

        def action_scaler(actions):
            resc_actions = (actions - orig_low) / (orig_high - orig_low) * 2 - 1
            return np.clip(resc_actions, -1 + safety_margin, 1 - safety_margin)
    else:
        action_scaler = lambda x: x

    transitions = []
    prev_actions = collections.deque(maxlen=action_queuesize)
    current_states = collections.deque(maxlen=frame_stack)

    for _ in range(action_queuesize):
        prev_action = np.zeros_like(traj['actions'][0])
        prev_actions.append(prev_action)

    for _ in range(frame_stack):
        state = traj['observations'][0]['state']
        current_states.append(state)

    last_observation = None
    last_action = None

    for i in range(len(traj['observations'])):
        obs = dict()
        if len(image_keys) == 1:
            obs['image'] = traj['observations'][i][image_keys[0]]
            obs['image'] = _reshape_image(obs['image'])
        else:
            for k in image_keys:
                obs[k] = _reshape_image(traj['observations'][i][k])

        current_states.append(traj['observations'][i]['state'])
        obs['state'] = np.concatenate(current_states, axis=-1)
        if add_prev_actions:
            obs['prev_actions'] = np.concatenate(prev_actions, axis=-1)

        action = traj['actions'][i]
        action = action_scaler(action)
        prev_actions.append(action)

        if task_id_mapping is not None:
            num_tasks = len(task_id_mapping.keys())
            if num_tasks > 1:
                task_id_vec = np.zeros(num_tasks, np.float32)
                task_id_vec[task_id_mapping[traj['task_description']]] = 1
                obs['task_id'] = task_id_vec

        is_positive = is_positive_sample(traj, i, traj['task_description'], num_final_reward_steps)

        if is_positive:
            reward = 1.0
        else:
            reward = 0.0

        mask = (1-reward)

        transitions.append(dict(
                observations=obs,
                actions=action,
                rewards=reward * config.reward_scale + config.reward_shift,
                masks=mask,
                dones_float=float(i == len(traj['observations']) - 1),
                ))

    for i in range(len(transitions)-1):
        transitions[i]['next_observations'] = transitions[i+1]['observations']

    transitions[len(transitions)-1]['next_observations'] = transitions[len(transitions)-1]['observations']
    return transitions