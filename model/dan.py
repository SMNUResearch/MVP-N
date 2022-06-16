import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # bmm: inputs must be 3-D tensors each containing the same number of matrices.
        # inputs: [b * n * m] [b * m * p] -> [b * n * p]
        # can also use torch.matmul
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / self.temperature
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        outputs = torch.bmm(attention, v)

        return outputs, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.w_q = nn.Linear(self.d_model, self.num_heads * self.d_q)
        self.w_k = nn.Linear(self.d_model, self.num_heads * self.d_k)
        self.w_v = nn.Linear(self.d_model, self.num_heads * self.d_v)
        self.fc = nn.Linear(self.num_heads * self.d_v, self.d_model)
 
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_q)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_v)))
        nn.init.xavier_normal_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(np.power(self.d_k, 0.5), dropout=dropout)

    def forward(self, q, k, v):
        residual = q
        q = self.w_q(q).view(q.shape[0], self.num_heads, self.d_q)
        k = self.w_k(k).view(k.shape[0], self.num_heads, self.d_k)
        v = self.w_v(v).view(v.shape[0], self.num_heads, self.d_v)

        q = q.permute(1, 0, 2).contiguous()
        k = k.permute(1, 0, 2).contiguous()
        v = v.permute(1, 0, 2).contiguous()

        outputs, attention = self.attention(q, k, v)
        outputs = outputs.permute(1, 0, 2).contiguous()
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(outputs + residual)

        return outputs, attention

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Conv1d(d_in, d_hidden, 1)
        self.conv2 = nn.Conv1d(d_hidden, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        outputs = x.unsqueeze(0).transpose(1, 2)
        outputs = self.conv2(F.relu(self.conv1(outputs)))
        outputs = outputs.transpose(1, 2)
        outputs = self.dropout(outputs)
        outputs = outputs.squeeze(0)
        outputs = self.layer_norm(outputs + residual)

        return outputs

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, num_heads, d_inner, dropout):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, d_q, d_k, d_v, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x):
        y, attention = self.attention(x, x, x)
        y = self.ffn(y)

        return y, attention

class Encoder(nn.Module):
    def __init__(self, h, d_model, d_q, d_k, d_v, num_heads, d_inner, dropout):
        super(Encoder, self).__init__()
 
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_q, d_k, d_v, num_heads, d_inner, dropout) for _ in range(h)])

    def forward(self, x):
        attention_list = []
        outputs = x
        for encoder_layer in self.layer_stack:
            outputs, attention = encoder_layer(outputs)
            attention_list += [attention]

        return outputs, attention_list

class DAN(nn.Module):
    def __init__(self, model, h, feature_dim, num_heads, inner_dim, dropout):
        super(DAN, self).__init__()

        self.h = h
        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.classifier = model.net.fc
        self.d_view_feature = feature_dim
        self.num_heads = num_heads
        self.d_inner = inner_dim
        self.dropout = dropout
        self.encoder = Encoder(self.h, self.d_view_feature, self.d_view_feature, self.d_view_feature, self.d_view_feature, self.num_heads, self.d_inner, self.dropout)

    def forward(self, batch_size, max_num_views, num_views, x):
        y = self.extractor(x)
        k = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view(z.shape[0], -1)
            z, attention_list = self.encoder(z)
            z = torch.max(z, 0)[0]
            k.append(z)

        k = torch.stack(k) # batch_size * num_features

        return self.classifier(k), k

