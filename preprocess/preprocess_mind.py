import os
from urllib import request
import sys
import zipfile
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from tqdm import trange
from datetime import datetime
from collections import defaultdict

tqdm.pandas()


def get_user_seq(df):
    user_group = df.groupby('uid')
    seq_dict = {}
    for _, group in user_group:
        seq_dict[_] = set()
        for log in group['sids']:
            seq_dict[_].add(log)
    for user, seq_set in seq_dict.items():
        for seq in seq_set:
            if not pd.isna(seq):
                seq_list = seq.split()
                seq_dict[user] = seq_list
            else:
                seq_dict[user] = []

    return seq_dict


def filter_triplets(user2items):
    print('Filtering tripltes.')
    min_sc = 5
    min_uc = 5
    print('Min item : {}'.format(min_sc))
    print('Min user interaction: {}'.format(min_uc))
    if min_sc > 0:
        item_counts = defaultdict(int)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1

        items_to_remove = {item for item, count in item_counts.items() if count < 5}

        for user, items in user2items.items():
            user2items[user] = [item for item in items if item not in items_to_remove]
    if min_uc > 0:
        user2items = {k: v for k, v in user2items.items() if len(v) > 5}
    return user2items


def densify_index(user2items):
    print('Desifying index.')
    umap = {u: (i + 1) for i, u in enumerate(user2items.keys())}
    item_set = set()
    for items in user2items.values():
        for item in items:
            item_set.add(item)
    smap = {s: (i + 1) for i, s in enumerate(sorted(item_set))}
    user2items = {umap[user]: [smap[item] for item in items] for user, items in user2items.items()}
    return umap, smap, user2items


def split_dataset(user2items):
    print('Splitting: Leave One Out.')
    train, val, test = {}, {}, {}
    for user in user2items.keys():
        items = user2items[user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    return train, val, test


def data_preprocess():
    path_save_dir = os.getcwd() + '\\' + 'MIND'
    df = pd.read_csv(os.path.join(path_save_dir, 'behaviors.tsv'), sep='\t', header=None, usecols=[1, 3])
    df.columns = ['uid', 'sids']
    user2items = get_user_seq(df)
    user2items = filter_triplets(user2items)
    umap, smap, user2items = densify_index(user2items)
    train, val, test = split_dataset(user2items)
    data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}

    return data_all


def save_data(data_all):
    if not os.path.exists('..//datasets'):
        os.makedirs('..//datasets')
    path_save_dir = '..//datasets//MIND'
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    if not os.path.exists(os.path.join(path_save_dir, 'dataset_5_5.pkl')):
        with open(os.path.join(path_save_dir, 'dataset_5_5.pkl'), 'wb') as f:
            pickle.dump(data_all, f)
        print()
        print('Preprocess Data Saved.')
    else:
        print()
        print('Preprocess data already exist.')


def main():
    pd_dataset = data_preprocess()
    save_data(pd_dataset)


if __name__ == '__main__':
    main()
