import torch
import torch.nn as nn
import torch.nn.functional as F

class GVCNN(nn.Module):
    def __init__(self, model, M, architecture_name, image_size):
        super(GVCNN, self).__init__()

        self.M = M # group num
        self.architecture_name = architecture_name
        self.image_size = image_size
        if self.architecture_name == 'RESNET18':
            # top five conv layers; num_channels: 64;
            self.FCN = nn.Sequential(*list(model.net.children())[:5])
            # 224*224: 56; 256*256: 64;
            feature_size = self.image_size // 4
            self.FC = nn.Sequential(nn.Linear(64 * feature_size * feature_size, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Linear(64, 1))
            self.CNN = nn.Sequential(*list(model.net.children())[5:-1])
            self.classifier = model.net.fc

    def fc_raw(self, x):
        y = self.FC(x.view(x.shape[0], -1))

        return y

    def ceil_score(self, scores):
        n = len(scores)
        s = 0
        for i in range(0, n):
            s += torch.ceil(scores[i] * n)

        s = s / n

        return s

    def group_fusion(self, group_views, group_weights):
        # equation 4:
        z = sum(list(map(lambda a, b: a * b, group_weights, group_views))) / sum(group_weights)

        return z

    def group_scheme(self, final_views, discrimination_scores, M):
        interval = 1.0 / M
        view_group = [[] for i in range(M)]
        score_group = [[] for i in range(M)]
        for i in range(0, M):
            left = i * interval
            right = (i + 1) * interval
            for j in range(0, discrimination_scores.shape[0]):
                if (discrimination_scores[j] >= left) and (discrimination_scores[j] < right):
                    score_group[i].append(discrimination_scores[j])
                    view_group[i].append(final_views[j])

                if right == 1 and discrimination_scores[j] == right: # only single view
                    score_group[i].append(discrimination_scores[j])
                    view_group[i].append(final_views[j])

        # equation 3: group level descriptors
        group_views = [sum(views) / len(views) for views in view_group if len(views) > 0]
        # equation 2:
        group_weights = [self.ceil_score(scores) for scores in score_group if len(scores) > 0]
 
        return group_views, group_weights

    def forward(self, batch_size, max_num_views, num_views, x):
        k = []
        count = 0
        raw_views = self.FCN(x)
        final_views = self.CNN(raw_views)
        final_views = final_views.view(final_views.shape[0], -1)
        for i in range(0, batch_size):
            r = raw_views[count:(count + max_num_views), :, :, :]
            f = final_views[count:(count + max_num_views), :]
            count += max_num_views
            r = r[0:num_views[i], :, :, :]
            f = f[0:num_views[i], :]
            discrimination_scores = self.fc_raw(r)
            # equation 1
            discrimination_scores = torch.sigmoid(torch.log(torch.abs(discrimination_scores)))
            # If all discrimination scores are very small, it may cause some problem in params update. option: add a softmax to normalize the scores.
            discrimination_scores = F.softmax(discrimination_scores, dim=0)
            group_views, group_weights = self.group_scheme(f, discrimination_scores, self.M)
            z = self.group_fusion(group_views, group_weights)
            k.append(z)

        k = torch.stack(k) # batch_size * num_features

        return self.classifier(k), k

