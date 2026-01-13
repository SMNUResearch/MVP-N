import torch
from torch import nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_q, num_kv, num_qk_proj, num_v_proj, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(num_q, num_qk_proj, bias=False)
        self.k_proj = nn.Linear(num_kv, num_qk_proj, bias=False)
        self.v_proj = nn.Linear(num_kv, num_v_proj, bias=False)
        self.o_proj = nn.Linear(num_v_proj, num_q)
        self.temperature = (num_qk_proj // self.num_heads) ** 0.5

    def forward(self, x_q, x_kv):
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)
        q, k, v = (rearrange(x, 'b n (h c) -> (b h) n c', h=self.num_heads) for x in [q, k, v])
        attention = torch.einsum('b i c, b j c -> b i j', q, k) / self.temperature # [batch_size * num_heads, num_views, num_views]
        attention_scores = attention.softmax(dim=-1) # normalized function
        attention = self.dropout(attention_scores)
        out = torch.einsum('b i j, b j c -> b i c', attention, v)
        out = rearrange(out, '(b h) n c -> b n (h c)', h=self.num_heads)

        return self.o_proj(out), attention_scores

class SelfAttention(nn.Module):
    def __init__(self, num_heads, num_channels, num_qk_proj, num_v_proj, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(num_heads, num_channels, num_channels, num_qk_proj, num_v_proj, dropout)

    def forward(self, x):
        x = self.norm(x)

        return self.attention(x, x)

class SelfAttentionLayer(nn.Module):
    def __init__(self, num_heads, num_channels, num_qk_proj, num_v_proj, attention_dropout, mlp_dropout, widening_factor):
        super().__init__()

        self.attention = SelfAttention(num_heads, num_channels, num_qk_proj, num_v_proj, attention_dropout)
        self.attention_drop = nn.Dropout(attention_dropout)
        self.mlp_drop = nn.Dropout(mlp_dropout)
        self.mlp = nn.Sequential(nn.LayerNorm(num_channels), nn.Linear(num_channels, widening_factor * num_channels),
                                 nn.GELU(), nn.Linear(widening_factor * num_channels, num_channels))

    def forward(self, x):
        out, attention_scores = self.attention(x)
        out1 = self.attention_drop(out) + x
        out = self.mlp(out1)
        out = self.mlp_drop(out) + out1

        return out, attention_scores

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, num_channels, num_qk_proj, num_v_proj, attention_dropout, mlp_dropout, widening_factor):
        super().__init__()

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(SelfAttentionLayer(num_heads, num_channels, num_qk_proj, num_v_proj, attention_dropout, mlp_dropout, widening_factor))

    def forward(self, x):
        attention_scores_list = []
        for attention_layer in self.attention_layers:
            x, attention_scores = attention_layer(x)
            attention_scores_list.append(attention_scores)

        return x, attention_scores_list

class VSFormer(nn.Module):
    def __init__(self, model, feature_dim, num_layers, num_heads, attention_dropout, mlp_dropout, widening_factor):
        super().__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.classifier = model.net.fc
        self.encoder = TransformerEncoder(num_layers, num_heads, feature_dim, feature_dim, feature_dim, attention_dropout, mlp_dropout, widening_factor)

    def forward(self, batch_size, max_num_views, num_views, x, use_utilization=False):
        y = self.extractor(x)
        k = []
        u = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view(1, z.shape[0], -1)
            z, _ = self.encoder(z)
            z = z.squeeze(0)

            if use_utilization:
                utilization = torch.zeros(max_num_views)
                view_utilization = torch.max(z, 0)[1]
                for j in range(0, z.shape[1]):
                    utilization[view_utilization[j]] += 1

                utilization = utilization / torch.sum(utilization)
                u.append(utilization)

            z = torch.max(z, 0)[0]
            k.append(z)

        k = torch.stack(k) # batch_size * num_features

        if use_utilization:
            u = torch.stack(u)

            return self.classifier(k), k, u, y

        return self.classifier(k), k, y

