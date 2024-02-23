import glob

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

def get_toykitchen2_pickplace(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/*')
    tasks += glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2_room8052/*')
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle', 'flip',
                    'topple', 'open', 'close', 'light_switch', 'upright', 'pour','drying_rack', 'faucet', 'turn']
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    print(task_names)

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen2_pickplace_cardboardfence_reversible(dir):
    tasks = ['put_lid_on_pot_cardboardfence',
             'take_lid_off_pot_cardboardfence',
             'put_bowl_on_plate_cardboard_fence',
             'take_bowl_off_plate_cardboard_fence',
             'put_sushi_in_pot_cardboard_fence',
             'take_sushi_out_of_pot_cardboard_fence',
             'put_carrot_in_pot_cardboard_fence',
             'take_carrot_out_of_pot_cardboard_fence',
             ]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_toykitchen2_sushi_targetdomain(dir):
    tasks = ['put_sushi_in_pot_cardboard_fence', 'take_sushi_out_of_pot_cardboard_fence',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val




def get_toykitchen2_pickplace_cardboardfence_reversible_simple(dir):
    tasks = ['put_lid_on_pot_cardboardfence',
            #  'take_lid_off_pot_cardboardfence',
             'put_bowl_on_plate_cardboard_fence',
            #  'take_bowl_off_plate_cardboard_fence',
             'put_sushi_in_pot_cardboard_fence',
            #  'take_sushi_out_of_pot_cardboard_fence',
             'put_carrot_in_pot_cardboard_fence',
            #  'take_carrot_out_of_pot_cardboard_fence',
             ]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen2/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_pickplace_reversible(dir):
    tasks = ['put_corn_in_bowl_sink',
            'take_corn_out_of_bowl_sink',
            'put_spoon_in_bowl_sink',
            'take_spoon_out_of_bowl_sink',
             'put_cup_on_plate',
             'take_cup_off_plate'
             ]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_target_domain(dir):
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-15/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_new_target_domain(dir):
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',
             'take_pear_from_plate',
             'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink',
             'put_sweet_potato_in_bowl',
             'put_drumstick_on_plate',
             'put_cucumber_in_orange_pot',
             'put_carrot_in_pan',
             'put_big_corn_in_big_pot']
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_toykitchen6_knife_in_pot(dir):
    tasks = ['put_knife_into_pot',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_croissant_out_of_pot(dir):
    tasks = ['take_croissant_out_of_pot',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toyktichen6_pear_from_plate(dir):
    tasks = ['take_pear_from_plate',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_sweet_potato_on_plate(dir):
    tasks = ['put_sweet_potato_on_plate',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_sweet_potato_in_bowl(dir):
    tasks = ['put_sweet_potato_in_bowl',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toyktichen6_put_lime_in_pan_sink(dir):
    tasks = ['put_lime_in_pan_sink',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_put_drumstick_on_plate(dir):
    tasks = ['put_drumstick_on_plate',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_cucumber_in_orange_pot(dir):
    tasks = ['put_cucumber_in_orange_pot',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_carrot_in_pan(dir):
    tasks = ['put_carrot_in_pan',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_big_corn_in_big_pot(dir):
    tasks = ['put_big_corn_in_big_pot',]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val



def get_toykitchen6_new_target_domain_5_tasks(dir):
    tasks = ['put_knife_into_pot',
             'take_croissant_out_of_pot',
             'take_pear_from_plate',
             'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink'
    ]
            #  'put_sweet_potato_in_bowl',
            #  'put_drumstick_on_plate',
            #  'put_cucumber_in_orange_pot',
            #  'put_carrot_in_pan',
            #  'put_big_corn_in_big_pot']
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen6_new_target_domain_2_tasks(dir):
    tasks = [
        # 'put_knife_into_pot',
            #  'take_croissant_out_of_pot',
            #  'take_pear_from_plate',
            #  'put_sweet_potato_on_plate',
             'put_lime_in_pan_sink',
            #  'put_sweet_potato_in_bowl',
             'put_drumstick_on_plate',
    ]
            #  'put_cucumber_in_orange_pot',
            #  'put_carrot_in_pan',
            #  'put_big_corn_in_big_pot']
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/targetdomain_6-18/toykitchen6/{}/val/out.npy'.format(task) for task in tasks]
    return train, val



def get_toykitchen1_pickplace(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/*')
    exclude_list = ['test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever']
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-1] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_toykitchen1_pickplace_cardboardfence_reversible(dir):
    tasks = [
        'put_broccoli_in_pan_cardboardfence',
        'put_carrot_on_plate_cardboardfence',
        'take_broccoli_out_of_pan_cardboardfence',
        'take_carrot_off_plate_cardboardfence'
    ]
    train = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{}/train/out.npy'.format(task) for task in tasks]
    val = [dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{}/val/out.npy'.format(task) for task in tasks]
    return train, val


def get_all_pickplace(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack'
                    ]
    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-2:] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

def get_all_pickplace_exclude_tk6(dir):
    tasks = glob.glob(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')
    # print(dir + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/*/*')
    # print(tasks)
    exclude_list = ['zip', 'test', 'box', 'fold_cloth_in_half', 'put_cap_on_container', 'open_book', 'basil_bottle',
                    'test', 'box', 'basket', 'knob', 'open', 'close', 'flip', 'lever', 'topple', 'pour', 'drying_rack',
                    'tool_chest', 'laundry_machine']
    exclude_list += ['toykitchen6']

    tasks = exclude_tasks(tasks, exclude_list)
    task_names = [str.split(path,  '/')[-2:] for path in tasks]
    for name in task_names:
        print(name)
    print(len(task_names))

    train = ['{}/train/out.npy'.format(task) for task in tasks]
    val = ['{}/val/out.npy'.format(task) for task in tasks]
    return train, val

if __name__ == '__main__':
    get_toykitchen1_pickplace()
    # get_all_pickplace()
    # get_toykitchen2_pickplace()
    # get_toykitchen2_pickplace_cardboardfence_reversible()
