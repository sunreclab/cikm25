import torch.utils.data as data_utils
import torch
import numpy as np
import pandas as pd


class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1]]
        tokens = seq[:-1]
        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        for user_temp, seq_temp in self.u2seq.items():
            for star in range(len(seq_temp)-1):
                self.id_seq[idx] = seq_temp[:star+2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, item_num):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.item_num = item_num

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        labels = answer
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(labels)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.item_counts = args.item_count

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len, self.item_counts)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len, item_nums):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.item_nums = item_nums

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        answer = self.u2answer[user]
        labels = answer
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(labels)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.item_count = args.item_count

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len, self.item_count)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


def get_item_cate(dataset, device, smap):
   if dataset == 'KuaiRec':
        path_data = '../preprocess/' + dataset + '/item_categories.csv'
        cate_set = set()
        movie_cate_raw = {}

        df = pd.read_csv(path_data, encoding='ISO-8859-1')
        for index, line in df.iterrows():
            movieID = int(line.iloc[0])
            generes = line.iloc[1].replace('[', '').replace(']', '').split(', ')
            for g in generes:
                cate_set.add(g)
            if movieID in smap.keys():
                movie_cate_raw[movieID] = movie_cate_raw.get(movieID, [])
                movie_cate_raw[movieID].extend(generes)

        movie_cate = {}  # {item_id:cate_dis}
        cate_l = sorted(list(cate_set))
        cate_ind = {cate_l[i]: i for i in range(len(cate_l))}

        for movieID, m_cate in movie_cate_raw.items():
            item_cate_val = [0 for _ in range(len(cate_l))]
            for c in m_cate:
                ind = cate_ind[c]
                item_cate_val[ind] = 1.0 / len(m_cate)
            movie_cate[movieID] = item_cate_val

        labels_list = [movie_cate[key] for key in movie_cate.keys()]

        labels_array = np.array(labels_list)  # shape: (item_num, 31)
        cate_mat = torch.tensor(labels_array, dtype=torch.float32, device=device)
        return cate_mat

   elif dataset == 'MIND':
        path_data = '../preprocess/' + dataset + '/news.tsv'
        cate_set = set()
        movie_cate_raw = {}

        df = pd.read_csv(path_data, encoding='ISO-8859-1', sep='\t', header=None)
        for index, line in df.iterrows():
            if line.iloc[0] in smap.keys():
                movieID = smap[line.iloc[0]]
            else:
                continue
            generes = line.iloc[1]

            cate_set.add(generes)
            movie_cate_raw[movieID] = movie_cate_raw.get(movieID, [])
            movie_cate_raw[movieID] = generes

        movie_cate = {}  # {item_id:cate_dis}
        cate_l = sorted(list(cate_set))
        cate_ind = {cate_l[i]: i for i in range(len(cate_l))}

        for movieID, m_cate in movie_cate_raw.items():
            item_cate_val = [0 for _ in range(len(cate_l))]
            ind = cate_ind[m_cate]
            item_cate_val[ind] = 1.0
            movie_cate[movieID] = item_cate_val

        labels_list = [movie_cate[key] for key in sorted(movie_cate.keys())]

        labels_array = np.array(labels_list)  # shape: (item_num, 15)
        cate_mat = torch.tensor(labels_array, dtype=torch.float32, device=device)
        return cate_mat

   elif dataset == 'Tenrec':
        path_data = '../preprocess/' + dataset + '/QB-article.csv'
        cate_set = set()
        movie_cate_raw = {}
        with open(path_data, 'r', encoding='ISO-8859-1') as fr:
            lines = fr.readlines()[1:]
            for line in lines:
                ele = line.strip('\n').split(',')
                movieID = int(ele[1])
                genere = ele[13]
                cate_set.add(genere)
                if movieID in smap.keys():
                    movie_cate_raw[movieID] = movie_cate_raw.get(movieID, [])
                    if genere not in movie_cate_raw[movieID]:
                        movie_cate_raw[movieID].append(genere)

        movie_cate = {}  # {item_id:cate_dis}
        cate_l = sorted(list(cate_set))
        cate_ind = {cate_l[i]: i for i in range(len(cate_l))}

        for movieID, m_cate in movie_cate_raw.items():
            item_cate_val = [0 for _ in range(len(cate_l))]
            for c in m_cate:
                ind = cate_ind[c]
                item_cate_val[ind] = 1.0 / len(m_cate)
            movie_cate[movieID] = item_cate_val

        labels_list = [movie_cate[key] for key in movie_cate.keys()]

        labels_array = np.array(labels_list)  # shape: (item_num, 42)
        cate_mat = torch.tensor(labels_array, dtype=torch.float32, device=device)
        return cate_mat
