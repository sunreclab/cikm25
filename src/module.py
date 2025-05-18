import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Disentangel_mask_soft(nn.Module):
    def __init__(self, ):
        super(Disentangel_mask_soft, self).__init__()
        self.gate_linear = nn.Linear(1, 2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.gate_linear.weight)

    def forward(self, seq_rep, seqs, mask_threshold, mask_inputs):
        alpha_score = torch.matmul(seqs, seq_rep.unsqueeze(-1))
        masks = F.gumbel_softmax(self.gate_linear((alpha_score - mask_threshold)), hard=True)[:, :, 0]
        return masks * mask_inputs, (1-masks) * mask_inputs


class Disentangel_mask_hard(nn.Module):
    def __init__(self, device, hidden_size):
        super(Disentangel_mask_hard, self).__init__()
        self.device = device
        # self.seq_linear = nn.Linear(hidden_size, hidden_size)
        # self.seqs_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, seq_rep, seqs, mask_threshold, mask_inputs):
        # seq_rep = torch.relu(self.seq_linear(seq_rep))
        # seqs = torch.relu(self.seqs_linear(seqs))
        seqs_norm = seqs/seqs.norm(dim=-1)[:, :, None]
        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        alpha_score = torch.matmul(seqs_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1) - mask_threshold
        

        mask_one = torch.where(alpha_score > 0, torch.ones_like(alpha_score).to(self.device), alpha_score)
        mask_onehot = torch.where(mask_one < 0, torch.zeros_like(alpha_score).to(self.device), mask_one)
        mask_onehot = mask_onehot - alpha_score.detach() + alpha_score
        return mask_onehot * mask_inputs, (1-mask_onehot) * mask_inputs


class Seq_mask_last_k(nn.Module):
    def __init__(self, hidden_size, k):
        super(Seq_mask_last_k, self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_k.weight)

    def forward(self, input):
        """
        :param input: B x L x H
        :return: B x H
        """
        if self.k != 1:
            seq_rep = self.linear_k(torch.mean(input[:, -self.k:, :], dim=1))
        else:
            seq_rep = self.linear_k(input[:, -1, :])
        return seq_rep


class Seq_mask_kth(nn.Module):
    def __init__(self, hidden_size, k):
        super(Seq_mask_kth, self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_k.weight)

    def forward(self, input):
        """
        :param input: B x L x H
        :return: B x H
        """
        if self.k != 1:
            seq_rep = self.linear_k(input[:, -self.k, :])
        else:
            seq_rep = self.linear_k(input[:, -1, :])
        return seq_rep


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Trend_interest_transforemer_block(nn.Module):
    def __init__(self, args):
        super(Trend_interest_transforemer_block, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class Hui_trend(nn.Module):
    def __init__(self):
        super(Hui_trend, self).__init__()

    def forward(self, seq_rep, embs):
        return seq_rep.unsqueeze(1) * embs.unsqueeze(0)


class MLP_diversity_rep(nn.Module):
    def __init__(self, hidden_size):
        super(MLP_diversity_rep, self).__init__()
        self.linear_w_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_3 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_4 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_4_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_5 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_5_1 = nn.Linear(hidden_size, hidden_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_w_1.weight)
        nn.init.xavier_normal_(self.linear_w_2.weight)
        nn.init.xavier_normal_(self.linear_w_3.weight)
        nn.init.xavier_normal_(self.linear_w_4.weight)
        nn.init.xavier_normal_(self.linear_w_4_1.weight)
        nn.init.xavier_normal_(self.linear_w_5.weight)
        nn.init.xavier_normal_(self.linear_w_5_1.weight)

    def forward(self, inputs):
        out = torch.relu(self.linear_w_1(inputs))
        out = torch.relu(self.linear_w_2(out))
        out = torch.relu(self.linear_w_3(out))
        out1 = torch.relu(self.linear_w_4(out))
        out1 = torch.relu(self.linear_w_5(out1))
        out2 = torch.relu(self.linear_w_4_1(out))
        out2 = torch.relu(self.linear_w_5_1(out2))
        return out1 + inputs, out2 + inputs
        # return out1 , out2


class Hui_diversity(nn.Module):
    def __init__(self, hidden_size, num_items):
        super(Hui_diversity, self).__init__()
        self.linear_w = nn.Linear(hidden_size, hidden_size)
        self.prediction_layer = nn.Linear(hidden_size, num_items-1)

    def forward(self, reps, masks):
        return torch.sigmoid(self.prediction_layer(self.linear_w(reps))) * masks


class Cross_rep(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(Cross_rep, self).__init__()
        self.cross_hidden_size = hidden_size // 2 + max_len
        self.total_rep_size = hidden_size * 2

        self.linear_1 = nn.Linear(self.cross_hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

        self.linear_3 = nn.Linear(self.cross_hidden_size, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)

        self.linear_w_1 = nn.Linear(self.total_rep_size, hidden_size)
        self.linear_w_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_w_3 = nn.Linear(hidden_size, 1)

    def forward(self, trend_item, trend_cat, div_item, div_cat):
        
        cross_hidden_one = torch.cat((trend_cat, div_item), dim=-1)
        cross_hidden_one = self.linear_1(cross_hidden_one)
        cross_hidden_one = self.linear_2(cross_hidden_one)

        cross_hidden_two = torch.cat((trend_item, div_cat), dim=-1)
        cross_hidden_two = self.linear_3(cross_hidden_two)
        cross_hidden_two = self.linear_4(cross_hidden_two)

        total_rep = torch.cat((cross_hidden_one, cross_hidden_two), dim=-1)

        out = torch.relu(self.linear_w_1(total_rep))
        out = torch.relu(self.linear_w_2(out))
        total_score = torch.relu(self.linear_w_3(out)).squeeze(-1)

        return total_score
