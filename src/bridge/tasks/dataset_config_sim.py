import os
import glob
import numpy as np

target_task = 'PutBallintoBowl-v0'
target_task_interfering = 'task_1_0'
filter_bin = ('task_3', '_5')
target_task_bin = 'task_3_5'
target_task_bin_easy = 'task_0_2'
target_task_bin_single = 'task_0_1'
target_task_bin_single_rev = 'task_1_0'


def get_data_binsort_stored_2obj(dir):
    train = [f'{dir}/minibullet/0811_addednoise_2obj_2000_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/train/out.npy']
    val = [f'{dir}/minibullet/0811_addednoise_2obj_2000_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/val/out.npy']
    return train, val

def get_data_binsort_2obj_easy(dir):
    train=[f'{dir}/minibullet/0604_multitask_binsort/{target_task_bin_easy}/BinSort-v0/train/out.npy']
    val=[f'{dir}/minibullet/0604_multitask_binsort/{target_task_bin_easy}/BinSort-v0/val/out.npy']
    return train, val

def get_data_binsort_stored_3obj(dir):
    train = [f'{dir}/minibullet/0802_addednoise_3obj_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/train/out.npy']
    val = [f'{dir}/minibullet/0802_addednoise_3obj_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/val/out.npy']
    return train, val

def get_data_binsort_stored_4obj(dir):
    train = [f'{dir}/minibullet/0802_addednoise_4obj_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/train/out.npy']
    val = [f'{dir}/minibullet/0802_addednoise_4obj_binsortneutralmultstored/{target_task_bin_single}/BinSort-v0/val/out.npy']
    return train, val

def get_multi_object_in_bowl_data(dir):
    paths = os.listdir(os.environ['DATA'] + '/minibullet/multitask_pickplacedata_noise0.1')

    paths = filter_tasks(paths)

    train = [f'{dir}/minibullet/multitask_pickplacedata_noise0.1/{task}/train/out.npy' for task in paths]
    val = [f'{dir}/minibullet/multitask_pickplacedata_noise0.1/{task}/val/out.npy' for task in paths]
    return train, val

def get_multi_object_in_bowl_data_interfering(dir):
    paths = os.listdir(dir + '/minibullet/0602_multitask_interfering')
    paths = filter_tasks(paths, target_task_interfering)

    train = [f'{dir}/minibullet/0602_multitask_interfering/{task}/PickPlaceInterferingDistractors-v0/train/out.npy' for task in paths]
    val = [f'{dir}/minibullet/0602_multitask_interfering/{task}/PickPlaceInterferingDistractors-v0/val/out.npy' for task in paths]
    
    for x in train:
        assert os.path.exists(os.environ['DATA'] + x), f'{os.environ["DATA"] + x} does not exist'
    
    return train, val

def get_data_binsort(dir, num_single_bin_tasks=8, num_multi_bin_tasks=5):
    subfolder = '0611_multitask_binsort_singleobj'
    paths = os.listdir(dir + f'/minibullet/{subfolder}')
    train_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSortSingleBin-v0/train/out.npy' for task in paths[:num_single_bin_tasks]]
    val_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSortSingleBin-v0/val/out.npy' for task in paths[:num_single_bin_tasks]]
    print(train_p1)
    
    subfolder = '0611_multitask_binsort'
    paths = os.listdir(dir + f'/minibullet/{subfolder}')
    paths = filter_tasks(paths, filter_bin[0])
    paths = filter_tasks(paths, filter_bin[1])
    
    train_p2 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/train/out.npy' for task in paths[:num_multi_bin_tasks]]
    val_p2 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/val/out.npy' for task in paths[:num_multi_bin_tasks]]
    
    train, val = train_p1 + train_p2, val_p1 + val_p2
    
    for x in train:
        assert os.path.exists(x), f'{x} does not exist'
    
    return train, val


def get_data_binsort_neg(dir, num_single_bin_tasks=8, num_multi_bin_tasks=5):
    subfolder = '0611_multitask_binsort_singleobj'
    paths = os.listdir(os.environ['DATA'] + f'/minibullet/{subfolder}')

    train_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSortSingleBin-v0/train/out.npy' for task in paths[:num_single_bin_tasks]]
    val_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSortSingleBin-v0/val/out.npy' for task in paths[:num_single_bin_tasks]]
    print(train_p1)
    
    subfolder = '0611_multitask_binsort'
    paths = os.listdir(dir + f'/minibullet/{subfolder}')
    paths = filter_tasks(paths, filter_bin[0])
    paths = filter_tasks(paths, filter_bin[1])
    
    train_p2 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/train/out.npy' for task in paths[:num_multi_bin_tasks]]
    val_p2 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/val/out.npy' for task in paths[:num_multi_bin_tasks]]
    
    subfolder = '0604_multitask_binsort'
    paths = os.listdir(dir + f'/minibullet/{subfolder}')
    paths = filter_tasks(paths, filter_bin[0])
    paths = filter_tasks(paths, filter_bin[1])
    
    train_p3 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/train/out.npy' for task in paths[:num_multi_bin_tasks]]
    val_p3 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/val/out.npy' for task in paths[:num_multi_bin_tasks]]
    
    train, val = train_p1 + train_p2 + train_p3, val_p1 + val_p2 + val_p3
    
    for x in train:
        assert os.path.exists(os.environ['DATA'] + x), f'{os.environ["DATA"] + x} does not exist'
    
    return train, val

# def get_data_binsort_single(num_single_bin_tasks=10):
#     subfolder = '06013_multitask_binsortneutral'
#     paths = os.listdir(os.environ['DATA'] + f'/minibullet/{subfolder}')
#     subtasks = list(set(paths[:num_single_bin_tasks] + [target_task_bin_single]))
#     train_p1 = [f'/minibullet/{subfolder}/{task}/BinSort-v0/train/out.npy' for task in subtasks]
#     val_p1 = [f'/minibullet/{subfolder}/{task}/BinSort-v0/val/out.npy' for task in subtasks]
    
#     train, val = train_p1, val_p1 
    
#     for x in train:
#         assert os.path.exists(os.environ['DATA'] + x), f'{os.environ["DATA"] + x} does not exist'
    
#     return train, val

def get_data_binsort_single(dir, num_single_bin_tasks=10):
    subfolder = '0619_multitask_binsortneutral'
    paths = os.listdir(os.environ['DATA'] + f'/minibullet/{subfolder}')
    np.random.shuffle(paths)
    
    subtasks = list(set(paths[:num_single_bin_tasks] + [target_task_bin_single, target_task_bin_single_rev]))
    train_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/train/out.npy' for task in subtasks]
    val_p1 = [f'{dir}/minibullet/{subfolder}/{task}/BinSort-v0/val/out.npy' for task in subtasks]
    
    train, val = train_p1, val_p1 
    
    for x in train:
        assert os.path.exists(os.environ['DATA'] + x), f'{os.environ["DATA"] + x} does not exist'
    
    return train, val

def get_data_binsort_single_target(dir):
    train = [dir + binsort_task_single]
    val =  [dir + binsort_task_single_val]
    return train, val


def get_data_ball_target(dir, num_demo=10):
    train = [f'{dir}/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/train/out{num_demo}.npy']
    val =  [f'{dir}/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/val/out.npy']
    return train, val


def filter_tasks(paths, target_task=target_task):
    new_paths = []
    for path in paths:
        if not target_task in path:
            new_paths.append(path)
    return new_paths


interfering_task=f'/minibullet/0602_multitask_interfering/{target_task_interfering}/PickPlaceInterferingDistractors-v0/train/out.npy'
interfering_task_val=f'/minibullet/0602_multitask_interfering/{target_task_interfering}/PickPlaceInterferingDistractors-v0/val/out.npy'

binsort_task=f'/minibullet/0604_multitask_binsort/{target_task_bin}/BinSort-v0/train/out.npy'
binsort_task_val=f'/minibullet/0604_multitask_binsort/{target_task_bin}/BinSort-v0/val/out.npy'


binsort_task_single=f'/minibullet/06013_finetuned_binsort_buffer/{target_task_bin_single}/BinSort-v0/train/out.npy'
binsort_task_single_val=f'/minibullet/06013_finetuned_binsort_buffer/{target_task_bin_single}/BinSort-v0/val/out.npy'

put_ball_in_bowl = '/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_val = '/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/val/out.npy'

put_ball_in_bowl_delay1 = '/minibullet/pickplacedata_noise0.1_delay1/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_val_delay1 = '/minibullet/pickplacedata_noise0.1_delay1/PutBallintoBowl-v0/val/out.npy'

put_ball_in_bowl_neg = '/minibullet/pickplacedata_noise0.1_failonly250/PutBallintoBowl-v0/train/out.npy'
put_ball_in_bowl_neg_val = '/minibullet/pickplacedata_noise0.1_failonly250/PutBallintoBowl-v0/val/out.npy'

if __name__ == '__main__':
    os.environ['DATA'] = '/nfs/kun2/users/asap7772/bridge_data_exps/sim_data'
    print(get_data_binsort())
    print(len(get_data_binsort()[0]))

