import torch
from torch import nn

class MVFN(nn.Module):
    def __init__(self, model, feature_dim):
        super().__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(self.feature_dim * 2, model.net.fc.out_features, bias=True)

    def forward(self, batch_size, max_num_views, num_views, x, use_utilization=False):
        y = self.extractor(x)
        k = []
        u = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view(z.shape[0], -1)
            z_pair = torch.cat([z[1:, :], z[0:1, :]], dim=0)
            pair_feature = torch.cat([z, z_pair], dim=1)

            if use_utilization:
                utilization = torch.zeros(max_num_views)
                view_utilization = torch.max(z, 0)[1]
                for j in range(0, z.shape[1]):
                    utilization[view_utilization[j]] += 1

                utilization = utilization / torch.sum(utilization)
                u.append(utilization)

            # remove neuron-wise correlation-regularized network layer: convergence issues
            z = torch.max(pair_feature, 0)[0]
            k.append(z)

        k = torch.stack(k)

        if use_utilization:
            u = torch.stack(u)

            return self.classifier(k), k, u, y

        return self.classifier(k), k, y

