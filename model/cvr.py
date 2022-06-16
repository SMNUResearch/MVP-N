import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils as tool

def sinkhorn(dot, max_iter):
    n, in_size, out_size = dot.shape
    K = dot
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)

    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))

    return K

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        out = self.dropout2(out)
        out = x + out

        return out

class PositionWiseFeedForwardBN(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super(PositionWiseFeedForwardBN, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.bn = nn.BatchNorm1d(d_hidden)
 
    def forward(self, x):
        out = self.fc1(x).transpose(1, 2)
        out = F.relu(self.bn(out).transpose(1, 2))
        out = x + self.fc2(out)

        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, h):
        super(ScaledDotProductAttention, self).__init__()

        self.fc_q = nn.Linear(d_model, h * d_q)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, q, k, v):
        q = self.fc_q(q).view(q.shape[0], q.shape[1], self.h, self.d_q).permute(0, 2, 1, 3).contiguous()
        k = self.fc_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).permute(0, 2, 3, 1).contiguous()
        v = self.fc_v(v).view(v.shape[0], v.shape[1], self.h, self.d_v).permute(0, 2, 1, 3).contiguous()
        attention = torch.matmul(q, k) / np.sqrt(self.d_k)
        attention = F.softmax(attention, dim=3)
        out = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.fc(out)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(d_model, d_q, d_k, d_v, num_heads)

    def forward(self, q, k, v):
        out = self.attention(q, k, v)
        out = self.dropout(out)
        out = self.layer_norm(q + out)
 
        return out

class MultiHeadAttentionBN(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, num_heads, dropout=0.1):
        super(MultiHeadAttentionBN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(d_model, d_q, d_k, d_v, num_heads)

    def forward(self, q, k, v):
        out = self.attention(q, k, v)
        out = self.dropout(out)
        out = q + out

        return out

class OTKernel(nn.Module):
    def __init__(self, in_dim, out_size, norm_eps, heads, eps, max_iter):
        super(OTKernel, self).__init__()

        self.in_dim = in_dim
        self.out_size = out_size
        self.norm_eps = norm_eps
        self.heads = heads
        self.eps = eps
        self.max_iter = max_iter

        self.weight = nn.Parameter(torch.Tensor(heads, out_size, in_dim))
        self.reset_parameter()
        nn.init.xavier_normal_(self.weight)

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.out_size)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def get_attention(self, x):
        K = torch.tensordot(tool.normalize(x, self.norm_eps), tool.normalize(self.weight, self.norm_eps), dims=[[-1], [-1]]) # [batch_size, view_num, heads, out_size]
        K = K.permute(0, 2, 1, 3)
        K = K.reshape(-1, K.shape[2], K.shape[3])
        K = torch.exp(K / self.eps)
        K = sinkhorn(K, self.max_iter)
        K = K.reshape(x.shape[0], self.heads, K.shape[1], K.shape[2])
        K = K.permute(0, 3, 1, 2).contiguous()

        return K

    def forward(self, x):
        batch_size = x.shape[0]
        attention_weight = self.get_attention(x)
        output = torch.bmm(attention_weight.view(batch_size, self.out_size * self.heads, -1), x)
        output = output.reshape(batch_size, self.out_size, -1)

        return output

class CVR(nn.Module):
    def __init__(self, model, K, feature_dim, num_heads, inner_dim, norm_eps, otk_heads, otk_eps, otk_max_iter, dropout, coord_dim):
        super(CVR, self).__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.classifier = model.net.fc
        self.K = K
        self.d_view_feature = feature_dim
        self.num_heads = num_heads
        self.d_hidden = inner_dim
        self.norm_eps = norm_eps
        self.otk_heads = otk_heads
        self.otk_eps = otk_eps
        self.otk_max_iter = otk_max_iter
        self.dropout = dropout
        self.coord_dim = coord_dim
        self.M = 2 ** self.K

        self.MHA1 = MultiHeadAttention(self.d_view_feature, self.d_view_feature, self.d_view_feature, self.d_view_feature, self.num_heads, self.dropout)
        self.ff1 = PositionWiseFeedForward(self.d_view_feature, self.d_hidden, self.dropout)
        self.otk_layer = OTKernel(self.d_view_feature, self.M, self.norm_eps, self.otk_heads, self.otk_eps, self.otk_max_iter)
        self.ff2 = PositionWiseFeedForwardBN(self.d_view_feature, self.d_hidden, self.dropout)
        self.coord_encoder = nn.Sequential(nn.Linear(self.d_view_feature, self.coord_dim), nn.ReLU(), nn.Linear(self.coord_dim, self.K))
        self.coord_decoder = nn.Sequential(nn.Linear(self.K, self.coord_dim), nn.ReLU(), nn.Linear(self.coord_dim, self.d_view_feature))
        self.MHA2 = MultiHeadAttentionBN(self.d_view_feature, self.d_view_feature, self.d_view_feature, self.d_view_feature, self.num_heads, self.dropout)
        self.ff3 = PositionWiseFeedForwardBN(self.d_view_feature, self.d_hidden, self.dropout)
 
    def forward(self, batch_size, max_num_views, num_views, x):
        y = self.extractor(x)
        k = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view(z.shape[0], -1)
            z = z.unsqueeze(0)
            out0 = self.MHA1(z, z, z)
            out0 = self.ff1(out0)
            out1 = self.otk_layer(out0)
            out1 = self.ff2(out1)
            pos0 = tool.normalize(self.coord_encoder(out1), self.norm_eps)
            pos1 = self.coord_decoder(pos0)
            out2 = self.MHA2(out1 + pos1, out1 + pos1, out1)
            out2 = self.ff3(out2)
            pooled_view = out2.mean(1)
            k.append(pooled_view)

        k = torch.cat(k) # batch_size * num_features

        if self.training:
            return self.classifier(k), k, pos0
        else:
            return self.classifier(k), k

