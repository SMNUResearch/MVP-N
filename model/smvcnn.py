import torch
import torch.nn as nn
import torch.nn.functional as F

class SMVCNN(nn.Module):
    def __init__(self, model, feature_dim, d):
        super(SMVCNN, self).__init__()

        self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        self.feature_dim = feature_dim
        self.d = d
        self.classifier = nn.Linear(self.d, model.net.fc.out_features, bias=True)
        self.downsampling = nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.upsampling = nn.ConvTranspose2d(self.feature_dim // 2, self.feature_dim, kernel_size=3, stride=2, padding=1, bias=True)
        self.embed = nn.Conv2d(self.feature_dim, self.d, kernel_size=1, stride=1, padding=0)

    def forward(self, batch_size, max_num_views, num_views, x):
        y = self.extractor(x)
        k = []
        count = 0
        for i in range(0, batch_size):
            z = y[count:(count + max_num_views), :, :, :]
            count += max_num_views
            z = z[0:num_views[i], :, :, :]
            attention = self.upsampling(self.relu(self.downsampling(z)))
            z = z.view((int(z.shape[0] / num_views[i]), num_views[i], z.shape[-3], z.shape[-2], z.shape[-1]))
            B, V, C, H, W = z.shape
            attention = F.softmax(attention.view(B, V, C, H, W), dim=1)
            z = torch.sum(attention * z, dim=1)
            z = self.embed(z)
            z = z.view(z.shape[0], -1)
            # remove l2 normalization layer: convergence issues
            k.append(z)

        k = torch.cat(k)

        return self.classifier(k), k

