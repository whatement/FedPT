import copy
import random

import numpy as np
from collections import Counter


def split_dataset(args, train_dataset, test_dataset):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_user_groups, test_user_groups = {}, {}

    # sample training data amongst users
    if args.partition == "homo":
        train_user_groups = split_by_iid(train_dataset, args.num_clients, args.num_classes)
        test_user_groups = split_by_iid(test_dataset, args.num_clients, args.num_classes)

    # Sample Non-IID user data
    elif args.partition == "dirichlet":
        train_user_groups = split_by_dirichlet(train_dataset, args.num_clients, args.num_classes, args.dirichlet_alpha)
        test_user_groups = split_by_iid(test_dataset, args.num_clients, args.num_classes)

    train_data_distribution_dicts = show_clients_data_distribution(train_dataset, train_user_groups, args.num_classes)
    test_data_distribution_dicts = show_clients_data_distribution(test_dataset, test_user_groups, args.num_classes)

    return train_user_groups, test_user_groups, train_data_distribution_dicts


def classify_label(dataset, num_classes: int):
    label_list = [[] for _ in range(num_classes)]
    for index, datum in enumerate(dataset):
        label_list[datum[1]].append(index)
    return label_list


def show_clients_data_distribution(dataset, dict_split: dict, num_classes):
    list_per_client = []
    clients_indices = list(dict_split.values())
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[round(idx)][1]
            nums_data[label] += 1
        list_per_client.append(nums_data)
        print(f'{client + 1}: {nums_data}')

    return list_per_client


def split_by_iid(dataset, num_clients: int, num_classes: int):
    dict_users = {}

    list_label_index = classify_label(dataset, num_classes)
    data_set = [set() for i in range(num_clients)]

    for i in range(len(list_label_index)):
        num_items = int(len(list_label_index[i]) / num_clients)
        for j in range(num_clients):
            new_set = set(np.random.choice(list_label_index[i], num_items,
                                           replace=False))
            old_set = copy.deepcopy(data_set[j])
            data_set[j] = old_set | new_set
            assert len(data_set[j]) == len(old_set) + len(new_set)
            list_label_index[i] = list(set(list_label_index[i]) - new_set)

    for k in range(num_clients):
        dict_users[k] = data_set[k]

    return dict_users

def split_by_dirichlet(train_dataset, num_clients, num_classes, dirichlet_alpha):
    train_dict_users = {}

    train_list_label_index = classify_label(train_dataset, num_classes)
    train_client_idcs = [[] for _ in range(num_clients)]

    # mini test
    i = 0
    while True:
        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients, num_classes)
        proportions /= proportions.sum(axis=1, keepdims=True)
        # Check if each client has data from at least min_samples_per_class different classes
        
        all_clients_have_min_samples = True
        for j in range(num_clients):
            if np.sort(proportions[:, j])[-2] * len(train_list_label_index[0]) < 1:
                all_clients_have_min_samples = False

        if all_clients_have_min_samples:
            break
        else:
            i = i + 1
            print("Try to generate the Dirichlet distribution, Try {}!".format(i))
    
    for c, fracs in zip(train_list_label_index, proportions):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            train_client_idcs[i] += [idcs]
    train_client_idcs = [np.concatenate(idcs) for idcs in train_client_idcs]

    for k in range(num_clients):
        train_dict_users[k] = set(train_client_idcs[k])

    return train_dict_users