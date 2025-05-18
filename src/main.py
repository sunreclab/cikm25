import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import pickle
from utils import Data_Train, Data_Val, Data_Test
from model import Dual_Disentangle
from trainer import model_train
import os
import logging
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='KuaiRec', help='Dataset name: KuaiRec, Tenrec, MIND')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=2025, help='Random seed')
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--item_count', type=int, default=3065, help='The total number of items in dataset')
parser.add_argument('--position_embedding_flag', type=str, default=True, help='Position embedding switch')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden Size')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of Trend interests transformer blocks')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of embedding')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'Adagrad'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
parser.add_argument('--training_flag', type=bool, default=True, help='Inference or Training, True for Training, otherwise')
parser.add_argument('--disentangle_para', type=lambda x: {k:float(v) for k,v in (i.split(':') for i in x.split(','))}, default='ci_adver:0.5,cd_adver:0.5', help='Parameter for representation disentanglement')
parser.add_argument("--num_groups", type=int, default=31, help="item group number in the dataset, must change for each dataset")
args = parser.parse_args()
print(args)


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)

if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset)
if not os.path.exists('model_dict/' + args.dataset):
    os.makedirs('model_dict/' + args.dataset)


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


fix_random_seed_as(args.random_seed)


def model_save(model, dataset, results):
    path_model = 'model_dict/' + dataset + '/' + str(results) + '_' + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())) + '.pt'
    torch.save(model.state_dict(), path_model)


def main():
    path_data = '../datasets/' + args.dataset + '/dataset_5_5.pkl'
    data_raw = pickle.load(open(path_data, 'rb'))
    args.item_count = len(data_raw['smap'])
    
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    model = Dual_Disentangle(args, data_raw['smap'])
    if args.training_flag:
        test_results, model_best = model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger, data_raw['smap'])
        model_save(model_best, args.dataset, test_results['Recall@10'])

    print(args)


if __name__ == '__main__':
    main()
