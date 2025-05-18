import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Disentangel_mask_hard, Disentangel_mask_soft, Seq_mask_last_k, Seq_mask_kth,  Trend_interest_transforemer_block, MLP_diversity_rep, Hui_trend, Hui_diversity, Cross_rep
from utils import get_item_cate


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Cate_predict_loss(nn.Module):
    def __init__(self, args, subseq):
        super(Cate_predict_loss, self).__init__()
        self.num_groups = args.num_groups
        if subseq == 'trend':
            self.hidden_size = args.hidden_size // 2
            # self.hidden_size = args.hidden_size
        else :
            self.hidden_size = args.max_len
        self.cate_pred_layer = nn.Linear(self.hidden_size, self.num_groups, bias=False)

        # criterion
        self.adverse_criterion = nn.BCEWithLogitsLoss()

    def forward(self, in_emb, cate_dist, is_inverse):
        if is_inverse == 1:
            in_emb.register_hook(lambda grad: -grad)

        cate_pred_logits = self.cate_pred_layer(in_emb)
        cate_dist = cate_dist.unsqueeze(0).expand(cate_pred_logits.shape[0], -1, -1)
        loss = self.adverse_criterion(input=cate_pred_logits, target=cate_dist)

        return loss


class Total_loss(nn.Module):
    def __init__(self, args, item_cate):
        super(Total_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cate_predict_loss_trend = Cate_predict_loss(args, 'trend')
        self.cate_predict_loss_diversity = Cate_predict_loss(args, 'diversity')
        self.item_cate = item_cate
        self.disentangle_para = args.disentangle_para

    def forward(self, pred_total, target,  user_emb_item_trend, user_emb_cat_trend, user_emb_item_div, user_emb_cat_div):
        # loss for accuracy
        t_loss = self.ce_loss(pred_total, target)

        # rep disentangle loss for trend_subseq
        ## ui_adversary_loss
        ui_trend_adver_loss = self.cate_predict_loss_trend(user_emb_item_trend, self.item_cate, is_inverse=1)
        ui_trend_adver_loss = self.disentangle_para['ci_adver'] * ui_trend_adver_loss
        ## uc_adversary_loss
        uc_trend_addi_loss = self.cate_predict_loss_trend(user_emb_cat_trend, self.item_cate, is_inverse=0)
        uc_trend_addi_loss = self.disentangle_para['cd_adver'] * uc_trend_addi_loss

        # rep disentangle loss for div_subseq
        ## ui_adversary_loss
        ui_div_adver_loss = self.cate_predict_loss_diversity(user_emb_item_div, self.item_cate, is_inverse=1)
        ui_div_adver_loss = self.disentangle_para['ci_adver'] * ui_div_adver_loss
        ## uc_adversary_loss
        uc_div_addi_loss = self.cate_predict_loss_diversity(user_emb_cat_div, self.item_cate, is_inverse=0)
        uc_div_addi_loss = self.disentangle_para['cd_adver'] * uc_div_addi_loss

        trend_seq_disentangel_loss = ui_trend_adver_loss + uc_trend_addi_loss
        div_seq_disentangel_loss = ui_div_adver_loss + uc_div_addi_loss

        total_loss = t_loss + trend_seq_disentangel_loss + div_seq_disentangel_loss
        return total_loss


class Dual_Disentangle(nn.Module):
    def __init__(self, args, smap):
        super(Dual_Disentangle, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.item_num = args.item_count+1
        self.max_len = args.max_len
        self.position_embedding_flag = args.position_embedding_flag
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_trend_rep = LayerNorm(self.hidden_size)
        self.norm_diversity_rep = LayerNorm(self.hidden_size)
        self.item_cate = get_item_cate(args.dataset, args.device, smap)

        self.seq_rep_mask_k = Seq_mask_last_k(self.hidden_size, k=1)
        self.disentangle_mask_hard = Disentangel_mask_hard(self.device, self.hidden_size)
        self.mask_threshold = 0.2

        self.transformer_trend_interest = Trend_interest_transforemer_block(args)
        # self.trend_seq_rep_mask_k = Seq_mask_last_k(self.hidden_size, k=1)
        self.hui_trend = Hui_trend()

        self.mlp_diversity_rep = MLP_diversity_rep(self.hidden_size)
        self.hui_diversity = Hui_diversity(self.hidden_size, self.item_num)

        self.cross_rep = Cross_rep(self.hidden_size, self.max_len)
        self.total_loss = Total_loss(args, self.item_cate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.position_embedding.weight)

    def forward(self, inputs):
        if self.position_embedding_flag:
            emb = self.item_embedding(inputs)
            pos_embedding = self.position_embedding.weight.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
            emb = self.embed_dropout(emb + pos_embedding)
        else:
            emb = self.embed_dropout(self.item_embedding(inputs))
            
        mask_input = (inputs > 0).float()
        rep_seq = self.seq_rep_mask_k(emb)
        mask_trend, mask_diversity = self.disentangle_mask_hard(rep_seq, emb, self.mask_threshold, mask_input)
     
        trend_emb = emb * mask_trend.unsqueeze(-1)
        diversity_emb = emb * mask_diversity.unsqueeze(-1)

        trend_rep = self.transformer_trend_interest(trend_emb, mask_input.unsqueeze(1).repeat(1, inputs.shape[1], 1).unsqueeze(1))
        
        trend_rep = self.dropout(self.norm_trend_rep(trend_rep))
        # trend_seq_rep = self.trend_seq_rep_mask_k(trend_rep)
        trend_seq_rep = trend_rep[:, -1, :]

        hui_trend = self.hui_trend(trend_seq_rep, self.item_embedding.weight[1:])
        user_emb_item_trend, user_emb_cate_trend = torch.split(hui_trend, self.hidden_size // 2, dim=2)

        diversity_rep_item, diversity_rep_cate = self.mlp_diversity_rep(emb)
        diversity_rep_item = self.dropout(self.norm_diversity_rep(diversity_rep_item))
        diversity_rep_cate = self.dropout(self.norm_diversity_rep(diversity_rep_cate))

        user_emb_item_div = self.hui_diversity(diversity_rep_item, mask_diversity.unsqueeze(-1))
        user_emb_cate_div = self.hui_diversity(diversity_rep_cate, mask_diversity.unsqueeze(-1))
        user_emb_item_div, user_emb_cate_div = user_emb_item_div.transpose(1, 2), user_emb_cate_div.transpose(1, 2)

        scores_final = self.cross_rep(user_emb_item_trend, user_emb_cate_trend, user_emb_item_div, user_emb_cate_div)
        return scores_final, user_emb_item_trend, user_emb_cate_trend, user_emb_item_div, user_emb_cate_div

