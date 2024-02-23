import glob
import os
import sys
import numpy as np

def exclude_tasks(paths, excluded_tasks):
    new_paths = []
    for d in paths:
        reject = False
        for exdir in excluded_tasks:
            if exdir in d:
                # print('excluding', d)
                reject = True
                break
        if not reject:
            new_paths.append(d)
    return new_paths

def include_tasks(paths, included_tasks):
    new_paths = []
    for d in paths:
        accept = False
        for exdir in included_tasks:
            if exdir in d:
                accept = True
                break
        if accept:
            new_paths.append(d)
    return new_paths

def get_epic_pickplace(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/epic100/*')
    exclude_list = ['bin', 'rinse', 'door', 'cupboard',  'drawer','fridge', 'hand', 'wash', 'cut', 'oven', 'stir',  'squeeze', 'tap', 'knife']
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    print(task_names)

def get_epic_knife(dir):
    tasks = ['knife-pick-up']
    train = [dir + f'robonetv2/toykitchen_numpy_shifted/epic100_bridgeform/epic100/{task}/train/out.npy' for task in tasks]
    val = [dir + f'robonetv2/toykitchen_numpy_shifted/epic100_bridgeform/epic100/{task}/val/out.npy' for task in tasks]
    return train, val

def get_epic_door(dir):
    tasks = ['cupboard-open', 'drawer-open', 'fridge-open', 'door-open', 'cupboard-close', 'drawer-close', 'fridge-close', 'door-close']
    train = [dir + f'robonetv2/toykitchen_numpy_shifted/epic100_bridgeform/epic100/{task}/train/out.npy' for task in tasks]
    val = [dir + f'robonetv2/toykitchen_numpy_shifted/epic100_bridgeform/epic100/{task}/val/out.npy' for task in tasks]
    return train, val



