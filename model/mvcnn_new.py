import torch
from torch import nn

class MVCNNNew(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.classifier = model.net.fc

    def forward(self, batch_size, max_num_views, num_views, x, use_utilization=False):
        y = self.extractor(x)
        k = []
        u = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view((int(z.shape[0] / num_views[i]), num_views[i], z.shape[-3], z.shape[-2], z.shape[-1]))

            if use_utilization:
                utilization = torch.zeros(max_num_views)
                view_utilization = torch.max(z, 1)[1].view(z.shape[0], -1).squeeze(0)
                for j in range(0, z.shape[-3]):
                    utilization[view_utilization[j]] += 1

                utilization = utilization / torch.sum(utilization)
                u.append(utilization)

            # view pooling
            z = torch.max(z, 1)[0].view(z.shape[0], -1) # shape: 1 * 512
            k.append(z)

        k = torch.cat(k) # batch_size * num_features

        if use_utilization:
            u = torch.stack(u)

            return self.classifier(k), k, u, y

        return self.classifier(k), k, y

