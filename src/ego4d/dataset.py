import sys
sys.path.append('../')

import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
from .data_utils import get_aliased_name
import time
from PIL import Image
from collections import defaultdict
import ml_collections

def get_ind(vid, index):
    im = Image.open(f'{vid}{index:06}.jpg')
    return transforms.ToTensor()(im)


# Data Loader for VIP
class GCSEgo4DDataset(IterableDataset):
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.1,
            'p_trajgoal': 0.7,
            'p_currgoal': 0.2,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'p_samegoal': 0.5,
            'intent_sametraj': True,
            'terminal': True,
            'max_distance': ml_collections.config_dict.placeholder(int),
        })

    def __init__(self, datapath='/nfs/nfs1/ego4d', batch_size=64, num_tasks=100, orig_manifest_path='/checkpoint/yixinlin/eaif/datasets/ego4d',
                 replace_manifest_path='/nfs/nfs1/ego4d', gamma=0.99, data_load_type='sarsa', text_as_task=False,
                 reward_scale=1, reward_shift=-1, terminal=True, p_randomgoal=.3, p_trajgoal=.5, p_currgoal=.2, p_samegoal=.5, intent_sametraj=False, max_distance=None):
        assert(datapath is not None)
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.orig_manifest_path = orig_manifest_path
        self.replace_manifest_path = replace_manifest_path
        self.data_load_type = data_load_type
        self.gamma = gamma
        self.text_as_task = text_as_task
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.terminal = terminal
        self.p_randomgoal = p_randomgoal
        self.p_trajgoal = p_trajgoal
        self.p_currgoal = p_currgoal
        self.p_samegoal = p_samegoal
        self.intent_sametraj = intent_sametraj
        self.max_distance = max_distance
        # Augmentations
        preprocess_transform = torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(128),
        )
        self.preprocess = lambda img: np.moveaxis(np.asarray(preprocess_transform(img)), -3, -1)

        # Load Data
        self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
        self.datalen = len(self.manifest)
        
        if 'task' not in self.manifest.columns:
            if self.text_as_task:
                assert 'text' in self.manifest.columns, "Must have either 'task' or 'text' column in manifest"
                self.manifest['task'] = self.manifest['text']
                print("WARNING: Using 'text' column as 'task' column would be deprecated in the future. Please use 'task' column instead.")
            else:
                # Create a task column with the name being taskless
                self.manifest['task'] = self.manifest['text'].apply(lambda x: "taskless") # defaults to 0 task id
        # add aliasing
        pre_alias_vc = self.manifest['task'].value_counts()
        self.manifest['task'] = self.manifest['task'].apply(get_aliased_name)
        vc = self.manifest['task'].value_counts()
        
        print("Original Task Counts: \n", pre_alias_vc)
        print("Aliased Task Counts: \n", vc)
        
        vc_dict = vc.to_dict()
        self.all_tasks = list(vc_dict.keys())
        assert len(self.all_tasks) <= self.num_tasks, f"Number of tasks ({len(self.all_tasks)}) must be less than or equal to num_tasks ({self.num_tasks}). Please increase num_tasks or reduce the number of tasks in the dataset {self.all_tasks}."
        task_ids = {t: i for i, t in enumerate(self.all_tasks)}
        
        print("Task IDs: ", task_ids)
        self.manifest['task_id'] = self.manifest['task'].apply(lambda x: task_ids[x])
        self.dataloader = iter(torch.utils.data.DataLoader(self, batch_size=self.batch_size, num_workers=96, pin_memory=True))
    
    def _sample_goal(self, state_frame_id, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        rand = np.random.rand()

        # If we are sampling a random or trajectory goal
        if rand < p_randomgoal + p_trajgoal:
            randomgoal = rand < p_randomgoal

            # If we are sampling a fully random goal, resample video id
            if randomgoal:
                goal_video_id = np.random.randint(0, self.datalen)
            # Otherwise use state's video id
            else:
                goal_video_id, _ = state_frame_id

            # Resample frame index
            vidlen = self.manifest.iloc[goal_video_id]['num_frames']
            _, state_frame_ind = state_frame_id
            min_frame_ind = 0 if randomgoal else state_frame_ind + 1
            max_frame_ind = min(state_frame_ind + self.max_distance + 1, vidlen) if self.max_distance else vidlen
            distance = np.random.rand()
            goal_frame_ind = int(round(min_frame_ind * distance + max_frame_ind * (1 - distance)))

        # Otherwise, we sample the current state
        else:
            goal_video_id, goal_frame_ind = state_frame_id
        vid = self.manifest.iloc[goal_video_id]['directory']
        vid = vid.replace(self.orig_manifest_path, self.replace_manifest_path)
        goal = get_ind(vid, goal_frame_ind)
        goal = self.preprocess(goal)
        return (goal_video_id, goal_frame_ind), goal

    def _sample(self, frame_id=None):
        # Sample a video
        if frame_id is None:
            video_id = np.random.randint(0, self.datalen)
            vidlen = self.manifest.iloc[video_id]["num_frames"]
            video_frame_ind = np.random.randint(0, vidlen)  
            frame_id = (video_id, video_frame_ind)
        else:
            video_id, video_frame_ind = frame_id
        video_metadata = self.manifest.iloc[video_id]
        vid = video_metadata["directory"]
        vid = vid.replace(self.orig_manifest_path, self.replace_manifest_path)
        text = video_metadata['text']
        
        # Sample goals
        if self.intent_sametraj:
            desired_goal_id, desired_goal = self._sample_goal(frame_id, p_randomgoal=0.0, p_trajgoal=1.0 - self.p_currgoal, p_currgoal=self.p_currgoal)
        else:
            desired_goal_id, desired_goal = self._sample_goal(frame_id)
        goal_id, goal = (desired_goal_id, desired_goal) if np.random.rand() < self.p_samegoal else self._sample_goal(frame_id)

        success = (frame_id == goal_id)
        desired_success = (frame_id == desired_goal_id)
        reward = float(success) * self.reward_scale + self.reward_shift
        desired_reward = float(desired_success) * self.reward_scale + self.reward_shift
        
        goal_video_id, goal_frame_ind = goal_id
        desired_video_id, _ = desired_goal_id

        if goal_video_id != desired_video_id:
            mc_return = self.reward_shift/(1 - self.gamma)
        else:
            mc_return = 10
            for _ in range(video_frame_ind, goal_frame_ind):
                mc_return = mc_return * self.gamma - self.reward_shift
            
        mask, desired_mask = (1.0 - success, 1.0 - desired_success) if self.terminal else (1.0, 1.0)
        
        task_id = np.zeros(self.num_tasks)
        task_id[video_metadata['task_id']] = 1
        
        default_action_dim=7
        action = np.zeros(default_action_dim).astype(float) 
            
        im = self.preprocess(get_ind(vid, video_frame_ind))
        next_im = self.preprocess(get_ind(vid, video_frame_ind + 1))

        return im, action, reward, desired_reward, next_im, mask, desired_mask, goal, desired_goal, task_id, text
    
    def sample(self, include_desired=True, include_text=False):
        ims, actions, rewards, desired_rewards, next_ims, masks, desired_masks, goals, desired_goals, task_ids, texts = next(self.dataloader)
        batch = {
            'observations': ims.numpy(),
            'actions': actions.numpy(),
            'rewards': rewards.numpy(),
            'desired_rewards': desired_rewards.numpy(),
            'next_observations': next_ims.numpy(),
            'masks': masks.numpy(),
            'desired_masks': desired_masks.numpy(),
            'goals': goals.numpy(),
            'desired_goals': desired_goals.numpy(),
            'task_ids': task_ids.numpy(),
            'texts': texts,
        }

        if not include_desired:
            batch.pop('desired_rewards')
            batch.pop('desired_masks')
            batch.pop('desired_goals')
        if not include_text:
            batch.pop('texts')
        
        return batch
    
    def get_trajectory(self, video_id=None, include_desired=False, include_text=False):
        if video_id is None:
            video_id = np.random.randint(0, self.datalen)
        vidlen = self.manifest.iloc[video_id]["num_frames"]
        batch = defaultdict(list)
        for frame_ind in range(vidlen):
            im, action, reward, desired_reward, next_im, mask, desired_mask, goal, desired_goal, task_id, text = self._sample((video_id, frame_ind))
            batch['observations'].append(im)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['desired_rewards'].append(desired_reward)
            batch['next_observations'].append(next_im)
            batch['masks'].append(mask)
            batch['desired_masks'].append(desired_mask)
            batch['goals'].append(goal)
            batch['desired_goals'].append(desired_goal)
            batch['task_ids'].append(task_id)
            batch['texts'].append(text)
        batch.update({k: np.array(v) for k, v in batch.items() if k != 'texts'})

        if not include_desired:
            batch.pop('desired_rewards')
            batch.pop('desired_masks')
            batch.pop('desired_goals')
        if not include_text:
            batch.pop('texts')
        return batch

    def __iter__(self):
        while True:
            yield self._sample()

if __name__ == "__main__":
    buffer = GCSEgo4DDataset()

    t0 = time.time()
    for i in range(1000):
        batch = buffer.sample()
        succ = (batch['rewards'][0] == 0)