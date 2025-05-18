import torch.nn as nn
import torch
import torch.optim as optim
import datetime
import numpy as np
import copy
from utils import get_item_cate


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer.lower() == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, initial_accumulator_value=1e-8)
    else:
        raise ValueError


def cal_recall(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def cal_diversity(label, predict, ks, item_cat):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    ce = []
    cc = []
    num_groups = item_cat.shape[1]
    new_item_cat = (item_cat != 0).float().numpy()
    for k in ks:
        cur_ce = 0.0
        cur_cc = 0.0
        for rec_item_list in topk_predict[:, :k]:
            cur_rec_cat = np.array([0] * num_groups, dtype=np.float32)
            for rec_item in rec_item_list:
                cur_rec_cat += np.array(new_item_cat[rec_item], dtype=np.float32)
            cur_rec_cat /= k
            entropy = -sum([e * np.log(e + 1e-12) for e in cur_rec_cat])
            cur_ce += entropy
            cur_cc += np.count_nonzero(cur_rec_cat)/num_groups
        cur_ce /= (topk_predict.shape[0])
        cur_cc /= (topk_predict.shape[0])
        ce.append(cur_ce)
        cc.append(cur_cc)
    return ce, cc


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def recalls_and_ndcgs_k(scores, labels, ks, item_cat):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_recall(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    ce, cc = cal_diversity(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks, item_cat.clone().detach().to('cpu'))
    for k, ndcg_temp, hr_temp, ce_temp, cc_temp in zip(ks, ndcg, hr, ce, cc):
        metrics['Recall@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
        metrics['CE@%d' % k] = ce_temp
        metrics['CC@%d' % k] = cc_temp
    return metrics  


def calculate_metrics(model, val_batch, metric_ks, item_cat):
    seqs, labels = val_batch
    labels = labels - 1
    scores, user_emb_item_trend, user_emb_cat_trend, user_emb_item_div, user_emb_cat_div = model(seqs)
    metrics = recalls_and_ndcgs_k(scores, labels, metric_ks, item_cat)
    return metrics


def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger, smap):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    item_cat = get_item_cate(args.dataset, device, smap)
    best_metrics_dict = {'Best_Recall@5': 0, 'Best_NDCG@5': 0, 'Best_Recall@10': 0, 'Best_NDCG@10': 0, 'Best_Recall@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_Recall@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_Recall@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_Recall@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()
        # lr_scheduler.step()
        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            score_final, user_emb_item_trend, user_emb_cat_trend, user_emb_item_div, user_emb_cat_div = model(train_batch[0])
            labels = train_batch[1].squeeze(-1)
            labels = labels - 1
            loss = model.total_loss(score_final, labels,  user_emb_item_trend, user_emb_cat_trend, user_emb_item_div, user_emb_cat_div)
            loss.backward()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            # quit()
            optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss.item()))
        print('start predicting: ', datetime.datetime.now())
        logger.info('start predicting: {}'.format(datetime.datetime.now()))
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'CE@5': [], 'CE@10': [], 'CE@20': [], 'CC@5': [], 'CC@10': [], 'CC@20': []}
            # metrics_dict_mean = {}
            for val_batch in val_data_loader:
                val_batch = [x.to(device) for x in val_batch]
                metrics = calculate_metrics(model, val_batch, metric_ks, item_cat)
                for k, v in metrics.items():
                    metrics_dict[k].append(v)
        
        for key_temp, values_temp in metrics_dict.items():
            if key_temp not in ['CE@5', 'CE@10', 'CE@20', 'CC@5', 'CC@10', 'CC@20']:
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    best_model = copy.deepcopy(model)
            else:
                values_mean = round(np.mean(values_temp), 4)

        if flag_update == 0:
            bad_count += 1
        else:
            print(best_metrics_dict)
            print(best_epoch)
            logger.info(best_metrics_dict)
            logger.info(best_epoch)
      
        if bad_count >= args.patience:
            break
        with torch.no_grad():
            test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'CE@5': [], 'CE@10': [], 'CE@20': [], 'CC@5': [], 'CC@10': [], 'CC@20': []}
            test_metrics_dict_mean = {}
            for test_batch in test_data_loader:
                test_batch = [x.to(device) for x in test_batch]

                metrics = calculate_metrics(best_model, test_batch, metric_ks, item_cat)
                for k, v in metrics.items():
                    test_metrics_dict[k].append(v)
        for key_temp, values_temp in test_metrics_dict.items():
            if key_temp not in ['CE@5', 'CE@10', 'CE@20', 'CC@5', 'CC@10', 'CC@20']:
                values_mean = round(np.mean(values_temp) * 100, 4)
            else:
                values_mean = round(np.mean(values_temp), 4)
            test_metrics_dict_mean[key_temp] = values_mean
        print('---------------------------Test------------------------------------------------------')
        logger.info('--------------------------Test------------------------------')
        print(test_metrics_dict_mean)
        logger.info(test_metrics_dict_mean)

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    with torch.no_grad():
        test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': [], 'CE@5': [], 'CE@10': [], 'CE@20': [], 'CC@5': [], 'CC@10': [], 'CC@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            metrics = calculate_metrics(best_model, test_batch, metric_ks, item_cat)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        if key_temp not in ['CE@5', 'CE@10', 'CE@20', 'CC@5', 'CC@10', 'CC@20']:
            values_mean = round(np.mean(values_temp) * 100, 4)
        else:
            values_mean = round(np.mean(values_temp), 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    return test_metrics_dict_mean, best_model
