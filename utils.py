import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from thop import profile

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def normalize(x, norm_eps):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    x = x / norm.clamp(min=norm_eps)

    return x 

def get_vert(k):
    M = 2 ** k
    vert = torch.zeros(M, k)
    for i in range(0, M):
        for j in range(0, k):
            divide = 2 ** (k - j - 1)
            index = i // divide
            if index % 2 == 0:
                vert[i][j] = 1
            else:
                vert[i][j] = -1

    return vert

def get_parameters(model_stage1, model_stage2):
    p1 = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
    p2 = sum(p.numel() for p in model_stage2.parameters() if p.requires_grad)

    return p1, p2

def get_FLOPs(model_stage1, model_stage2, inputs):
    f1 = profile(model_stage1, inputs=(inputs, ), verbose=False)[0]
    f2 = profile(model_stage2, inputs=(1, inputs.shape[0], [inputs.shape[0]], inputs, ), verbose=False)[0]

    return f1, f2

def get_time(model_stage1, model_stage2, inputs, repetition):
    model_stage1.eval()
    model_stage2.eval()
    # GPU WARM UP
    for i in range(10):
        outputs = model_stage1(inputs)

    t1 = np.zeros((repetition, 1))
    t2 = np.zeros((repetition, 1))

    with torch.no_grad():
        for i in range(0, repetition):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model_stage1(inputs)
            end.record()
            # wait for GPU synchronize
            torch.cuda.synchronize()
            current_time = start.elapsed_time(end)
            t1[i] = current_time

    with torch.no_grad():
        for i in range(0, repetition):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs, features = model_stage2(1, inputs.shape[0], [inputs.shape[0]], inputs)
            end.record()
            torch.cuda.synchronize()
            current_time = start.elapsed_time(end)
            t2[i] = current_time

    return np.mean(t1), np.std(t1), np.mean(t2), np.std(t2)

def plot_confusion_matrix(confusion_matrix, groups, mv_type, save=False):
    length = confusion_matrix.shape[0]
    plt.imshow(confusion_matrix, plt.cm.Reds)
    plt.colorbar()
    plt.ylabel('Ground Truths')
    plt.xlabel('Predictions')
    plt.title(mv_type)
    for i in range(0, len(groups)):
        start_point = int(groups[i][0])
        end_point = int(groups[i][len(groups[i]) - 1]) + 1
        plt.plot((start_point - 0.5, start_point - 0.5), (start_point - 0.5, end_point - 0.5), linewidth=1, color='blue')
        plt.plot((start_point - 0.5, end_point - 0.5), (start_point - 0.5, start_point - 0.5), linewidth=1, color='blue')
        plt.plot((end_point - 0.5, end_point - 0.5), (start_point - 0.5, end_point - 0.5), linewidth=1, color='blue')
        plt.plot((start_point - 0.5, end_point - 0.5), (end_point - 0.5, end_point - 0.5), linewidth=1, color='blue')

    if save == False:
        plt.show()
    else:
        plt.savefig(mv_type + '.pdf', dpi=1200, format='pdf', bbox_inches='tight', pad_inches=0)

class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        super(CosineDecayLR, self).__init__()

        self.optimizer = optimizer
        self.T_max = T_max
        self.lr_min = lr_min
        self.lr_max = lr_init
        self.warmup = warmup

    def step(self, t):
        if self.warmup and t < self.warmup:
            lr = self.lr_max / self.warmup * t
        else:
            T_max = self.T_max - self.warmup
            t = t - self.warmup
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t / T_max * np.pi))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

