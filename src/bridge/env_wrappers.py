import collections

import gym
import numpy as np
from gym.spaces import Box, Dict
import roboverse

class RescaleActions(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        act_high = np.ones(7)
        self.orig_low = env.action_space.low
        self.orig_high = env.action_space.high
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def step(self, action):
        if isinstance(action, dict):
            action['action'] = self.unnormalize_actions(action['action'])
        else:
            action = self.unnormalize_actions(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def rescale_actions(self, actions, safety_margin=0.01):
        """
        rescale xyz, and rotation actions to be within -1 and 1, then clip actions to stay within safety margin
        used when loading unnormalized actions into the replay buffer (the np files store unnormalized actions)
        """
        actions = actions.squeeze()
        assert actions.shape == (7,)

        resc_actions = (actions - self.orig_low) / (self.orig_high - self.orig_low) * 2 - 1
        return np.clip(resc_actions, -1 + safety_margin, 1 - safety_margin)

    def unnormalize_actions(self, actions):
        """
        rescale xyz, and rotation actions to be within the original environments bounds e.g. +-0.05 for xyz, +-0.25 for rotations
        """
        actions = actions.squeeze()
        assert actions.shape == (7,)
        actions_rescaled = (actions + 1) / 2 * (self.orig_high - self.orig_low) + self.orig_low

        if np.any(actions_rescaled > self.orig_high):
            print('action bounds violated: ', actions)
        if np.any(actions_rescaled < self.orig_low):
            print('action bounds violated: ', actions)
        return actions_rescaled

def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))

class Roboverse(gym.ObservationWrapper):

    def __init__(self, env, num_tasks=1, from_states=False, add_states=True):
        super().__init__(env)

        # Note that the previous action is multiplied by FLAGS.frame_stack to
        # account for the ability to pass in multiple previous actions in the
        # system. This is the case especially when the number of previous actions
        # is supposed to be many, when using multiple past frames
        obs_dict = {}
        self.from_states = from_states
        self.add_states = add_states
        if not from_states:
            obs_dict['image'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        if add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(10,), dtype=np.float32)
        if num_tasks > 1:
            obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)    
        self.observation_space = Dict(obs_dict)

    def observation(self, obs):
        out_dict = {}
        if 'image' in obs:
            out_dict['image'] = _process_image(obs['image'])[None]
        if 'state' in obs and (self.from_states or self.add_states):
            out_dict['state'] = obs['state'][None]
        return out_dict


class FrameStack(gym.Wrapper):

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        assert 'image' in self.observation_space.spaces
        pixel_obs_spaces = self.observation_space.spaces['image']

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis],
                        num_stack,
                        axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis],
                         num_stack,
                         axis=-1)
        new_pixel_obs_spaces = Box(low=low,
                                   high=high,
                                   dtype=pixel_obs_spaces.dtype)
        self.observation_space.spaces['image'] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append(obs['image'])
        obs['image'] = self.frames
        return obs

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs['image'])
        obs['image'] = self.frames
        return obs, reward, done, info

class PrevActionStack(gym.Wrapper):

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        prev_action_stack_spaces = Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

        self._env_dim = prev_action_stack_spaces.shape[0]

        low = np.repeat(prev_action_stack_spaces.low[..., np.newaxis],
                        num_stack,
                        axis=-1)
        high = np.repeat(prev_action_stack_spaces.high[..., np.newaxis],
                         num_stack,
                         axis=-1)
        new_action_stack_spaces = Box(low=low,
                                   high=high,
                                   dtype=prev_action_stack_spaces.dtype)
        self.observation_space.spaces['prev_action'] = new_action_stack_spaces

        self._action_frames = collections.deque(maxlen=num_stack)

    def reset(self):
        next_obs = self.env.reset()
        # At reset pass in all zeros previous actions
        for i in range(self._num_stack):
            self._action_frames.append(np.zeros(self._env_dim))
        next_obs['prev_action'] = self.action_frames[None]
        return next_obs

    @property
    def action_frames(self):
        return np.stack(self._action_frames, axis=-1)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        if isinstance(action, dict):
            action = action['action'].squeeze()
        self._action_frames.append(action)
        next_obs['prev_action'] = self.action_frames[None]
        return next_obs, reward, done, info

    def observation(self, observation):
        print ('Going through action stacking')
        return {
            'image': observation['image'],
            'state': observation['state'],
            'prev_action': observation['prev_action']
        }
        
        
def get_sim_env(dataset, random_initialization=False, num_bins=5):
    extra_kwargs = dict()
    
    if dataset == 'interfering':
        env_name = 'PickPlaceInterferingDistractors-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(1,0)
        )
    elif dataset in ['binsort', 'binsort_neg']:
        env_name = 'BinSort-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(3,5)
        )
        time_limit=100
    elif dataset in ['binsort_single', 'binsort_single_target', 'binsort_stored_2obj']:
        env_name = 'BinSort-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(0,1)
        )
        time_limit=100
    elif dataset == 'binsort_2obj_easy':
        env_name = 'BinSort-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(0,2)
        )
        time_limit=100    
    elif dataset == 'binsort_stored_3obj':
        env_name = 'BinSort-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(0,1,2)
        )
        time_limit=120
    elif dataset == 'binsort_stored_4obj':
        env_name = 'BinSort-v0'
        extra_kwargs=dict(
            specific_task_id=True,
            desired_task_id=(0,1,2,3)
        )
        time_limit=180
    elif dataset == 'ball_in_bowl':
        if random_initialization:
            env_name = 'PutBallintoBowlRandInit-v0'
            extra_kwargs=dict()
            time_limit=60 # extra time for the random init
            extra_kwargs=dict(
                random_intialization_bins=num_bins,
            )
        else:
            env_name = 'PutBallintoBowl-v0'
            time_limit=60
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")
    
    env = roboverse.make(env_name, transpose_image=True, **extra_kwargs)
    return env, time_limit
