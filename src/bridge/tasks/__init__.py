from .dataset_config_real import *
from .toykitchen_pickplace_dataset import *
from .get_door_openclose_data import *
from .epic_kitchens import *
from .dataset_config_sim import *

all_datasets = {
    'single_task': get_single_task,
    '11tasks': get_11tasks,
    'tk1_pickplace': get_toykitchen1_pickplace,
    'tk2_pickplace': get_toykitchen2_pickplace,
    'open_micro_single': get_single_task_openmicro,
    'openclose_all': get_openclose_all,
    'open_close': get_openclose_all,
    'openclose_exclude_tk1': get_openclose_exclude_tk1,
    # 'online_reaching_pixels': (online_reaching_pixels, online_reaching_pixels_val),
    # 'online_reaching_pixels_first100': (online_reaching_pixels_first100, online_reaching_pixels_val_first100),
    'toykitchen1_pickplace': get_toykitchen1_pickplace,
    'toykitchen2_pickplace': get_toykitchen2_pickplace,
    'all_pickplace': get_all_pickplace,
    'all_pickplace_except_tk6': get_all_pickplace_exclude_tk6,
    'toykitchen2_pickplace_simpler': get_toykitchen2_pickplace_cardboardfence_reversible_simple,
    'toykitchen6_knife_in_pot': get_toykitchen6_knife_in_pot,
    'toykitchen6_croissant_out_of_pot': get_toykitchen6_croissant_out_of_pot,
    'toykitchen6_pear_from_plate': get_toyktichen6_pear_from_plate,
    'toykitchen6_sweet_potato_on_plate': get_toykitchen6_put_sweet_potato_on_plate,
    'toykitchen6_sweet_potato_in_bowl': get_toykitchen6_put_sweet_potato_in_bowl,
    'toykitchen6_lime_in_pan_sink': get_toyktichen6_put_lime_in_pan_sink,
    'toykitchen6_drumstick_on_plate': get_toykitchen6_put_drumstick_on_plate,
    'toykitchen6_cucumber_in_pot': get_toykitchen6_cucumber_in_orange_pot,
    'toykitchen6_carrot_in_pan': get_toykitchen6_carrot_in_pan,
    'epic_openclose': get_epic_door,
    'epic_pickplace': get_epic_pickplace,
    'epic_knife': get_epic_knife,
    'ball_in_bowl': get_multi_object_in_bowl_data,
}

all_target_datasets = {
    'toykitchen2_pickplace_cardboardfence_reversible': get_toykitchen2_pickplace_cardboardfence_reversible,
    'toykitchen2_pickplace_simpler': get_toykitchen2_pickplace_cardboardfence_reversible_simple,
    'toykitchen6_pickplace_reversible': get_toykitchen6_pickplace_reversible,
    'toykitchen6_target_domain': get_toykitchen6_target_domain,
    'toykitchen6_new_target_domain': get_toykitchen6_new_target_domain,
    'toykitchen6_target_domain_two_tasks': get_toykitchen6_new_target_domain_2_tasks,
    'toykitchen6_target_domain_five_tasks': get_toykitchen6_new_target_domain_5_tasks,
    'toykitchen6_knife_in_pot': get_toykitchen6_knife_in_pot,
    'toykitchen6_croissant_out_of_pot': get_toykitchen6_croissant_out_of_pot,
    'toykitchen6_pear_from_plate': get_toyktichen6_pear_from_plate,
    'toykitchen6_sweet_potato_on_plate': get_toykitchen6_put_sweet_potato_on_plate,
    'toykitchen6_sweet_potato_in_bowl': get_toykitchen6_put_sweet_potato_in_bowl,
    'toykitchen6_lime_in_pan_sink': get_toyktichen6_put_lime_in_pan_sink,
    'toykitchen6_drumstick_on_plate': get_toykitchen6_put_drumstick_on_plate,
    'toykitchen6_cucumber_in_pot': get_toykitchen6_cucumber_in_orange_pot,
    'toykitchen6_carrot_in_pan': get_toykitchen6_carrot_in_pan,
    'toykitchen6_big_corn_in_big_pot': get_toykitchen6_big_corn_in_big_pot,
    'toykitchen1_pickplace_cardboardfence_reversible': get_toykitchen1_pickplace_cardboardfence_reversible,
    'toykitchen2_sushi_targetdomain': get_toykitchen2_sushi_targetdomain,
    'tk1_target_openmicrowave': tk1_targetdomain_openmicro_6,
    'target_ball_in_bowl': get_data_ball_target,
}


def get_tasks(dataset, target_dataset, dataset_directory):
    print('Looking for dataset: ', dataset, 'in directory: ', dataset_directory)
    assert dataset == '' or dataset in all_datasets
    assert target_dataset == '' or target_dataset in all_target_datasets

    if dataset != '':
        dataset_fn = all_datasets.get(dataset)
        train_tasks, eval_tasks = dataset_fn(dataset_directory)
    else:
        train_tasks, eval_tasks = [], []

    if target_dataset != '':
        target_dataset_fn = all_target_datasets.get(target_dataset)
        target_train_tasks, target_eval_tasks = target_dataset_fn(dataset_directory)
    else:
        target_train_tasks, target_eval_tasks = [], []
    return (train_tasks, eval_tasks), (target_train_tasks, target_eval_tasks)

def get_task_id_mapping(task_folders, task_aliasing_dict=None, index=-3):
    task_descriptions = set()
    for task_folder in task_folders:
        task_description = str.split(task_folder, '/')[index]
        if task_aliasing_dict and task_description in task_aliasing_dict:
            task_description = task_aliasing_dict[task_description]
        task_descriptions.add(task_description)
    task_descriptions = sorted(task_descriptions)
    task_dict = {task_descp: index for task_descp, index in
            zip(task_descriptions, range(len(task_descriptions)))}
    print ('Printing task descriptions..............')
    for idx, desc in task_dict.items():
        print (idx, ' : ', desc)
    print ('........................................')
    return task_dict