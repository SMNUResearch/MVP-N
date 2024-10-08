import torch
import numpy as np
import torch.nn as nn
import scipy.stats as stats
import torch.nn.functional as F

class LabelCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelCrossEntropy, self).__init__()

    def forward(self, outputs, targets):
        outputs_ls = F.log_softmax(outputs, dim=1)
        batch_size = outputs_ls.shape[0]
        loss = -(outputs_ls * targets).sum() / batch_size

        return loss

class KDLoss(nn.Module):
    def __init__(self, T, alpha):
        super(KDLoss, self).__init__()

        self.T = T
        self.alpha = alpha
        self.criterion = LabelCrossEntropy()

    def forward(self, outputs, targets, outputs_teacher):
        soft_targets = F.softmax(outputs_teacher / self.T, dim=1)
        loss_ce = self.criterion(outputs, targets)
        loss_soft = F.kl_div(F.log_softmax(outputs / self.T, dim=1), soft_targets, reduction='batchmean')
        loss = self.alpha * self.T * self.T * loss_soft + (1 - self.alpha) * loss_ce

        return loss

class SoftBootstrapping(nn.Module):
    def __init__(self, beta, checkpoint):
        super(SoftBootstrapping, self).__init__()

        self.beta = beta
        self.checkpoint = checkpoint
        self.criterion = LabelCrossEntropy()

    def forward(self, outputs, targets, epoch):
        if epoch < self.checkpoint:
            loss = self.criterion(outputs, targets)
        else:
            loss_ce = self.criterion(outputs, targets)
            predictions = F.softmax(outputs.detach(), dim=1)
            loss_soft = self.criterion(outputs, predictions)
            loss = self.beta * loss_ce + (1 - self.beta) * loss_soft

        return loss

class HardBootstrapping(nn.Module):
    def __init__(self, beta, checkpoint):
        super(HardBootstrapping, self).__init__()

        self.beta = beta
        self.checkpoint = checkpoint
        self.criterion = LabelCrossEntropy()

    def forward(self, outputs, targets, epoch):
        if epoch < self.checkpoint:
            loss = self.criterion(outputs, targets)
        else:
            batch_size = outputs.shape[0]
            loss_ce = self.criterion(outputs, targets)
            hard_targets = F.softmax(outputs.detach(), dim=1).argmax(dim=1).view(-1, 1) # shape: batch size * 1
            loss_hard = -(F.log_softmax(outputs, dim=1).gather(1, hard_targets)).sum() / batch_size
            loss = self.beta * loss_ce + (1 - self.beta) * loss_hard

        return loss

class LabelSmoothing(nn.Module):
    def __init__(self, smooth):
        super(LabelSmoothing, self).__init__()

        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs_ls = F.log_softmax(outputs, dim=1)
        batch_size, num_classes = outputs_ls.shape
        u = torch.ones_like(targets) / num_classes
        loss_ce = -(outputs_ls * targets).sum() / batch_size
        loss_u = -(outputs_ls * u).sum() / batch_size
        loss = (1 - self.smooth) * loss_ce + self.smooth * loss_u

        return loss

class BetaMixture(nn.Module):
    def __init__(self, alphas, betas, lambdas, max_loss_bound, min_loss_bound, loss_bound, resolution, max_iteration, avoid_zero_eps, EM_eps, nan_eps):
        super(BetaMixture, self).__init__()

        self.alphas = torch.tensor(alphas)
        self.betas = torch.tensor(betas)
        self.lambdas = torch.tensor(lambdas)
        self.max_loss_bound = max_loss_bound
        self.min_loss_bound = min_loss_bound
        self.loss_bound = loss_bound
        self.resolution = resolution
        self.max_iteration = max_iteration
        self.avoid_zero_eps = avoid_zero_eps
        self.EM_eps = EM_eps
        self.nan_eps = nan_eps

        self.lookup_table = torch.zeros(1, self.resolution)

    def loss_normalization(self, x, calculated=True, max_loss=0., min_loss=0.):
        loss = x.clone()
        if calculated == True:
            max_loss = np.percentile(loss, self.max_loss_bound)
            min_loss = np.percentile(loss, self.min_loss_bound)
            loss_normalized = loss[(loss <= max_loss) & (loss >= min_loss)]
        else:
            loss_normalized = loss

        loss_normalized = (loss_normalized - min_loss) / (max_loss - min_loss + self.avoid_zero_eps)
        loss_normalized[loss_normalized >= 1 - self.loss_bound] = 1 - self.loss_bound
        loss_normalized[loss_normalized <= self.loss_bound] = self.loss_bound

        return loss_normalized, max_loss, min_loss

    def check_lookup_table(self, x):
        loss_normalized = x.clone()
        loss = (loss_normalized * self.resolution).long()
        loss[loss < 0] = 0
        loss[loss >= self.resolution] = self.resolution - 1

        return self.lookup_table[loss]

    def update(self, loss, w):
        loss_mean = torch.sum(w * loss) / (torch.sum(w) + self.avoid_zero_eps)
        loss_variance = torch.sum(w * ((loss - loss_mean)**2) / (torch.sum(w) + self.avoid_zero_eps))
        alpha = loss_mean * (loss_mean * (1 - loss_mean) / (loss_variance + self.avoid_zero_eps) - 1)
        beta = alpha * (1 - loss_mean) / (loss_mean + self.avoid_zero_eps)

        return alpha, beta

    def expectation_maximization(self, x):
        loss = x.clone()
        # EM on beta distribution is unstable when x=0 or x=1.
        loss[loss >= 1 - self.EM_eps] = 1 - self.EM_eps
        loss[loss <= self.EM_eps] = self.EM_eps
        for i in range(self.max_iteration):
            # E step
            likelihood_clean = torch.from_numpy(stats.beta.pdf(loss.detach().numpy(), self.alphas[0], self.betas[0]))
            likelihood_noisy = torch.from_numpy(stats.beta.pdf(loss.detach().numpy(), self.alphas[1], self.betas[1]))
            r = torch.zeros(2, loss.shape[0])
            r[0] = self.lambdas[0] * likelihood_clean
            r[1] = self.lambdas[1] * likelihood_noisy
            r[r <= self.nan_eps] = self.nan_eps
            r /= r.sum(dim=0)
            # M step
            self.alphas[0], self.betas[0] = self.update(loss, r[0])
            self.alphas[1], self.betas[1] = self.update(loss, r[1])
            self.lambdas = r.sum(dim=1) / r.sum()

        self.create_lookup_table()

    def create_lookup_table(self):
        loss_table = torch.linspace(self.nan_eps, 1 - self.nan_eps, steps=self.resolution)
        self.lookup_table = self.posterior(loss_table)
        self.lookup_table[torch.argmax(self.lookup_table):] = self.lookup_table.max()

    def posterior(self, x):
        loss = x.clone()
        loss[loss >= 1 - self.EM_eps] = 1 - self.EM_eps
        loss[loss <= self.EM_eps] = self.EM_eps
        likelihood_clean = torch.from_numpy(stats.beta.pdf(loss.detach().cpu().numpy(), self.alphas[0], self.betas[0]))
        likelihood_noisy = torch.from_numpy(stats.beta.pdf(loss.detach().cpu().numpy(), self.alphas[1], self.betas[1]))
        probability = self.lambdas[0] * likelihood_clean + self.lambdas[1] * likelihood_noisy
        p = self.lambdas[0] * likelihood_clean / (probability + self.avoid_zero_eps)

        return p

class DynamicLabelCrossEntropy(nn.Module):
    def __init__(self):
        super(DynamicLabelCrossEntropy, self).__init__()

    def forward(self, outputs, targets, beta):
        outputs_ls = F.log_softmax(outputs, dim=1)
        batch_size = outputs_ls.shape[0]
        loss = -(beta * outputs_ls * targets).sum() / batch_size

        return loss

class DynamicSoftBootstrapping(nn.Module):
    def __init__(self):
        super(DynamicSoftBootstrapping, self).__init__()

        self.DCE = DynamicLabelCrossEntropy()

    def forward(self, outputs, targets, beta):
        predictions = F.softmax(outputs.detach(), dim=1)
        loss = self.DCE(outputs, targets, beta.view(-1, 1)) + self.DCE(outputs, predictions, (1 - beta).view(-1, 1))

        return loss

class DynamicHardBootstrapping(nn.Module):
    def __init__(self):
        super(DynamicHardBootstrapping, self).__init__()

        self.DCE = DynamicLabelCrossEntropy()

    def forward(self, outputs, targets, beta):
        batch_size = outputs.shape[0]
        hard_targets = F.softmax(outputs.detach(), dim=1).argmax(dim=1).view(-1, 1)
        loss_hard = -(F.log_softmax(outputs, dim=1).gather(1, hard_targets))
        loss = self.DCE(outputs, targets, beta.view(-1, 1)) + (((1 - beta).view(-1, 1)) * loss_hard).sum() / batch_size

        return loss

class DynamicBootstrapping(nn.Module):
    def __init__(self, beta_model, loss_bound, checkpoint, sv_type, device):
        super(DynamicBootstrapping, self).__init__()

        self.beta_model = beta_model
        self.loss_bound = loss_bound
        self.checkpoint = checkpoint
        self.sv_type = sv_type
        self.device = device

        if self.sv_type == 'DSB':
            self.criterion = DynamicSoftBootstrapping()
        elif self.sv_type == 'DHB':
            self.criterion = DynamicHardBootstrapping()

    def forward(self, outputs, targets, epoch, max_loss_epoch, min_loss_epoch):
        outputs_ls = F.log_softmax(outputs, dim=1)
        batch_size = outputs_ls.shape[0]
        loss_ce = -(outputs_ls * targets).sum(dim=1)
        if epoch < self.checkpoint:
            loss = loss_ce.sum() / batch_size
        else:
            loss_ce_norm, _, _ = self.beta_model.loss_normalization(loss_ce, calculated=False, max_loss=max_loss_epoch, min_loss=min_loss_epoch)
            beta_d = self.beta_model.check_lookup_table(loss_ce_norm)
            beta_d[beta_d <= self.loss_bound] = self.loss_bound
            beta_d[beta_d >= 1 - self.loss_bound] = 1 - self.loss_bound
            loss = self.criterion(outputs, targets, beta_d.type(torch.FloatTensor).to(self.device))

        return loss_ce, loss

    def update(self, loss_overall):
        loss_overall_norm, max_loss_epoch, min_loss_epoch = self.beta_model.loss_normalization(loss_overall, calculated=True, max_loss=0., min_loss=0.)
        self.beta_model.expectation_maximization(loss_overall_norm)

        return max_loss_epoch, min_loss_epoch

class LikelihoodRatioTest(nn.Module):
    def __init__(self, data_length, num_classes, device, delta_x, delta_y, retro_epoch, update_epoch, interval_epoch, every_n_epoch, soft_eps, update_eps, rho, flip_eps):
        super(LikelihoodRatioTest, self).__init__()

        self.data_length = data_length
        self.num_classes = num_classes
        self.device = device
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.retro_epoch = retro_epoch
        self.update_epoch = update_epoch
        self.interval_epoch = interval_epoch
        self.every_n_epoch = every_n_epoch
        self.soft_eps = soft_eps
        self.update_eps = update_eps
        self.rho = rho
        self.flip_eps = flip_eps

        self.A = 1.0 / self.num_classes * torch.ones(self.data_length, self.num_classes, self.num_classes, requires_grad=False).to(device) # correction matrix
        self.h = torch.zeros(self.data_length, self.num_classes)
        self.predict_soft_labels = torch.zeros(self.data_length, self.every_n_epoch, self.num_classes)
        self.criterion = LabelCrossEntropy()
        self.soft_labels = torch.zeros(self.data_length, self.num_classes).to(self.device)
        self.transform_targets = torch.zeros(self.data_length).long().to(self.device)

    def forward(self, outputs, targets, epoch, index):
        outputs_ls = F.log_softmax(outputs, dim=1)
        outputs_normal = F.softmax(outputs, dim=1)
        if epoch == 0: # get initial soft labels
            targets[targets >= 1 - self.soft_eps] = 1 - self.soft_eps
            targets[targets <= self.soft_eps] = self.soft_eps / self.num_classes
            self.soft_labels[index] = targets
            self.transform_targets[index] = torch.max(targets, 1)[1]

        if epoch < self.retro_epoch:
            loss = self.criterion(outputs, targets)
        else:
            # cross-entropy loss + retroactive loss
            A_batch = self.A[index].to(self.device)
            soft_labels_batch = self.soft_labels[index].to(self.device)
            loss = sum([-A_batch[i].matmul(soft_labels_batch[i].reshape(-1, 1).float()).t().matmul(outputs_ls[i]) for i in range(len(index))]) / len(index) + self.criterion(outputs, targets)

        if epoch in [self.retro_epoch - 1, self.retro_epoch - 1 + self.interval_epoch]:
            self.h[index] = outputs_ls.detach().cpu()

        if epoch >= (self.update_epoch - self.every_n_epoch):
            self.predict_soft_labels[index, epoch % self.every_n_epoch, :] = outputs_normal.detach().cpu()

        return loss

    def update(self, epoch):
        if epoch in [self.retro_epoch - 1, self.retro_epoch - 1 + self.interval_epoch]:
            unsolved = 0
            infeasible = 0
            with torch.no_grad():
                for i in range(self.data_length):
                    try:
                        h = self.h[i].reshape(-1, 1)
                        s = self.soft_labels[i].reshape(-1, 1)
                        A_temp = torch.ones(len(s), len(s)) * self.update_eps
                        A_temp[s.argmax(0)] = self.rho - self.update_eps / (len(s) - 1)
                        result = -((A_temp.matmul(s)).t()).matmul(h)
                    except:
                        unsolved += 1
                        continue

                    if (result == np.inf):
                        infeasible += 1
                    else:
                        self.A[i] = A_temp

        if epoch >= self.update_epoch: # apply LRT scheme
            y_tilde = self.transform_targets.clone().cpu()
            predict_soft_labels_bar = self.predict_soft_labels.mean(1)
            delta = self.delta_x + self.delta_y * max(epoch - self.update_epoch + 1, 0)
            for i in range(self.data_length):
                cond_1 = (not predict_soft_labels_bar[i].argmax() == y_tilde[i])
                cond_2 = (predict_soft_labels_bar[i].max() / predict_soft_labels_bar[i][y_tilde[i]] > delta)
                if cond_1 and cond_2:
                    y_tilde[i] = predict_soft_labels_bar[i].argmax()

            clean_soft_labels = torch.ones(self.data_length, self.num_classes) * self.flip_eps / (self.num_classes - 1)
            clean_soft_labels.scatter_(1, y_tilde.reshape(-1, 1), 1 - self.flip_eps)

            self.soft_labels = clean_soft_labels.clone().detach()
            self.transform_targets = clean_soft_labels.argmax(1).clone().detach()

class SelfAdaptiveTraining(nn.Module):
    def __init__(self, data_length, num_classes, device, checkpoint, alpha):
        super(SelfAdaptiveTraining, self).__init__()

        self.data_length = data_length
        self.num_classes = num_classes
        self.device = device
        self.checkpoint = checkpoint
        self.alpha = alpha

        self.criterion = LabelCrossEntropy()
        self.labels = torch.zeros(self.data_length, self.num_classes).to(self.device)

    def forward(self, outputs, targets, epoch, index):
        if epoch == 0: # get initial labels
            self.labels[index] = targets

        batch_labels = self.labels[index].to(self.device)
        if epoch < self.checkpoint:
            loss = self.criterion(outputs, batch_labels)
        else:
            confidence = F.softmax(outputs.detach(), dim=1)
            self.labels[index] = self.alpha * self.labels[index] + (1 - self.alpha) * confidence
            weights = self.labels[index].max(dim=1)[0]
            weights *= outputs.shape[0] / weights.sum()
            loss = torch.sum(-F.log_softmax(outputs, dim=1) * self.labels[index], dim=1)
            loss = (loss * weights).mean()

        return loss

class ProgressiveLabelCorrection(nn.Module):
    def __init__(self, data_length, num_classes, device, roll_window, warm_up, delta, step_size, ratio, max_delta):
        super(ProgressiveLabelCorrection, self).__init__()

        self.data_length = data_length
        self.num_classes = num_classes
        self.device = device
        self.roll_window = roll_window
        self.warm_up = warm_up
        self.delta = delta
        self.step_size = step_size
        self.ratio = ratio
        self.max_delta = max_delta

        self.criterion = LabelCrossEntropy()
        self.f_record = torch.zeros([self.roll_window, self.data_length, self.num_classes])
        self.labels = torch.zeros(self.data_length, self.num_classes).to(self.device)
        self.transform_targets = torch.zeros(self.data_length).long().to(self.device)

    def forward(self, outputs, targets, epoch, index):
        if epoch == 0: # get initial labels
            self.labels[index] = targets
            self.transform_targets[index] = torch.max(targets, 1)[1]

        batch_labels = self.labels[index].to(self.device)
        loss = self.criterion(outputs, batch_labels)
        self.f_record[epoch % self.roll_window, index] = F.softmax(outputs.detach().cpu(), dim=1)

        return loss

    def update(self, epoch):
        if epoch >= self.warm_up:
            f_x = self.f_record.mean(0)
            y_tilde = self.transform_targets.clone().cpu()
            y_corrected = self.lrt_correction(y_tilde, f_x)
            self.transform_targets = y_corrected.clone().detach()
            self.labels = torch.zeros(len(y_corrected), self.num_classes)
            y_corrected = y_corrected.unsqueeze(dim=1)
            self.labels.scatter_(dim=1, index=y_corrected, value=1)

    def lrt_correction(self, y_tilde, f_x):
        correction_count = 0
        y_noise = y_tilde.clone().detach()
        n = len(y_noise)
        f_m = f_x.max(1)[0]
        y_m = f_x.argmax(1)
        LR = []

        for i in range(n):
            LR.append(float(f_x[i][int(y_noise[i])] / f_m[i]))

        for i in range(n):
            if LR[i] < self.delta:
                y_noise[i] = y_m[i]
                correction_count += 1

        if correction_count < self.ratio * n:
            self.delta += self.step_size
            self.delta = min(self.delta, self.max_delta)

        return y_noise

class OnlineLabelSmoothing(nn.Module):
    def __init__(self, num_classes, device, alpha):
        super(OnlineLabelSmoothing, self).__init__()

        self.K = num_classes
        self.device = device
        self.alpha = alpha
        self.criterion = LabelCrossEntropy()
        # S: soft label matrix; Initialize S0
        self.S = torch.eye(self.K).to(device)
        # S_t: initialize S_t = 0 every epoch
        self.S_t = torch.zeros((self.K, self.K)).to(device)
        # count: normalization for each class
        self.count = torch.zeros((self.K, 1)).to(device)

    def forward(self, outputs, targets):
        target_label = torch.max(targets, 1)[1]
        prediction = torch.max(outputs, 1)[1]
        soft_targets = self.S[target_label]
        loss = self.alpha * self.criterion(outputs, targets) + (1 - self.alpha) * self.criterion(outputs, soft_targets)
        # utilize correct predicted scores
        outputs_prob = F.softmax(outputs.detach(), dim=1)
        batch_size = outputs.shape[0]
        correct_index = prediction.eq(target_label)
        for b in range(0, batch_size):
            if correct_index[b] == 1:
                class_index = target_label[b]
                self.S_t[class_index] += outputs_prob[b]
                self.count[class_index] += 1

        return loss

    def update(self):
        norm_index = torch.nonzero(self.count.squeeze(1))
        self.S[norm_index] = self.S_t[norm_index] / self.count[norm_index]

        nn.init.constant_(self.S_t, 0.)
        nn.init.constant_(self.count, 0.)

