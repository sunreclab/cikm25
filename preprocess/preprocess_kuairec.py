import os
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def make_implicit(df):
    print('Turning into implicit ratings.')
    min_watch_ratio = 2.0
    print('Min watch_ratio: {}'.format(min_watch_ratio))
    df = df[df['watch_ratio'] >= min_watch_ratio]
    return df


def filter_triplets(df):
    print('Filtering tripltes.')
    min_sc = 5
    min_uc = 5
    print('Min item : {}'.format(min_sc))
    print('Min user interaction: {}'.format(min_uc))
    if min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        df = df[df['sid'].isin(good_items)]
    if min_uc > 0:
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        df = df[df['uid'].isin(good_users)]
    return df


def densify_index(df):
    print('Desifying index.')
    umap = {u: (i + 1) for i, u in enumerate(sorted(set(df['uid'])))}
    smap = {s: (i + 1) for i, s in enumerate(sorted(set(df['sid'])))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    return df, umap, smap


def split_df(df, user_count):
    print('Splitting: Leave One Out.')
    user_group = df.groupby('uid')
    user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
    train, val, test = {}, {}, {}
    for user in range(1, user_count + 1):
        items = user2items[user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    return train, val, test


def data_preprocess():
    path_save_dir = os.getcwd() + '\\' + 'KuaiRec'
    df = pd.read_csv(os.path.join(path_save_dir, 'small_matrix.csv'), sep=',',
                     usecols=['user_id', 'video_id', 'timestamp', 'watch_ratio'])
    df.columns = ['uid', 'sid', 'timestamp', 'watch_ratio']
    df = make_implicit(df)
    df = filter_triplets(df)
    df, umap, smap = densify_index(df)
    train, val, test = split_df(df, len(umap))
    data_all = {'train': train, 'val': val, 'test': test, 'umap': umap, 'smap': smap}
    return data_all


def save_data(data_all):
    if not os.path.exists('..//datasets'):
        os.makedirs('..//datasets')
    path_save_dir = '..//datasets//KuaiRec'
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
