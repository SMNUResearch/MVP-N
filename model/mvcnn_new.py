import torch
import torch.nn as nn

class MVCNNNew(nn.Module):
    def __init__(self, model):
        super(MVCNNNew, self).__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.classifier = model.net.fc

    def forward(self, batch_size, max_num_views, num_views, x):
        y = self.extractor(x)
        k = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            z = z.view((int(z.shape[0] / num_views[i]), num_views[i], z.shape[-3], z.shape[-2], z.shape[-1]))
            # view pooling
            z = torch.max(z, 1)[0].view(z.shape[0], -1) # shape: 1 * 512
            k.append(z)

        k = torch.cat(k) # batch_size * num_features

        return self.classifier(k), k

