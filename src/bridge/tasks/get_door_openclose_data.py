import glob
import os

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

def get_openclose_all(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')

    include_list = ['open', 'close']
    exclude_list = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box']

    tasks = include_tasks(tasks, include_list)
    tasks = exclude_tasks(tasks, exclude_list)

    all_openclose_train = ['{}/train/out.npy'.format(task) for task in tasks]
    all_openclose_val = ['{}/val/out.npy'.format(task) for task in tasks]

    # print(all_openclose_train)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    # print('task_names', set(task_names))

    return all_openclose_train, all_openclose_val

def get_openclose_exclude_tk1(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')

    include_list = ['open', 'close']
    exclude_list = ['book', 'pick_up_closest_rainbow_Allen_key_set', 'box', 'toykitchen1']

    tasks = include_tasks(tasks, include_list)
    tasks = exclude_tasks(tasks, exclude_list)

    all_openclose_train = ['{}/train/out.npy'.format(task) for task in tasks]
    all_openclose_val = ['{}/val/out.npy'.format(task) for task in tasks]

    return all_openclose_train, all_openclose_val

def tk1_targetdomain_openmicro(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_closemicro(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/close_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_7_19_openmicro(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data_7-19/toykitchen1/open_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_7_19_closemicro(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data_7-19/toykitchen1/close_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val
    

def get_tk1_openmicro_failures(dir):
    tasks = [dir + '/robonetv2/extracted_online_data/selected_online_negatives/open_microwave_online_negatives',
            dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_negatives']

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def online_open_micro(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/online_data_extracted/open_only_run3/toykitchen1/open_microwave']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_openmicro_2(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_2']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_closemicro_2(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/close_microwave_2']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def tk1_targetdomain_openmicro_2_few_demo(dir, num_demo=25):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_2_few_demo']
    train = ['{}/train/out_{}demos.npy'.format(task, num_demo) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_openmicro_2_few_demo_with_negs(dir, num_demo=25):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_2_few_demo_with_neg']
    train = ['{}/train/out_{}demos_20negs.npy'.format(task, num_demo) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_openmicro_3(dir, num_demo=25):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_3']
    if num_demo is None or num_demo == 25:
        train = ['{}/train/out.npy'.format(task) for task in tasks]
    else:
        train = ['{}/train/out{}.npy'.format(task, num_demo) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_closemicro_3(dir, num_demo=25):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/close_microwave_3']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_openmicro_4(dir, num_demo=25):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_4']
    if num_demo is None or num_demo == 25:
        train = ['{}/train/out.npy'.format(task) for task in tasks]
    else:
        train = ['{}/train/out{}.npy'.format(task, num_demo) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def tk1_targetdomain_closemicro_4(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/close_microwave_4']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_openmicro_5(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_5']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def tk1_targetdomain_openmicro_6(dir, num_demo=None):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/open_microwave_6']
    if num_demo is None:
        train = ['{}/train/out.npy'.format(task) for task in tasks]
    else:
        train = ['{}/train/out{}.npy'.format(task, num_demo) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def tk1_targetdomain_closemicro_6(dir):
    tasks = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_data/toykitchen1/close_microwave_6']
    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val
