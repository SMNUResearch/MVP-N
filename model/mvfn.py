import torch
import torch.nn as nn

class MVFN(nn.Module):
    def __init__(self, model, feature_dim):
        super(MVFN, self).__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(self.feature_dim * 2, model.net.fc.out_features, bias=True)

    def forward(self, batch_size, max_num_views, num_views, x):
        y = self.extractor(x)
        k = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view(z.shape[0], -1)
            z_pair = torch.cat([z[1:, :], z[0:1, :]], dim=0)
            pair_feature = torch.cat([z, z_pair], dim=1)
            # remove neuron-wise correlation-regularized network layer: convergence issues
            z = torch.max(pair_feature, 0)[0]
            k.append(z)

        k = torch.stack(k)

        return self.classifier(k), k

