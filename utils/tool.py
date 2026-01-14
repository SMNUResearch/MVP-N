import random

import torch
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

    # GPU warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model_stage1(inputs)

    t1 = np.zeros(repetition, dtype=np.float32)
    t2 = np.zeros(repetition, dtype=np.float32)

    stream = torch.cuda.current_stream()

    with torch.no_grad():
        for i in range(repetition):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record(stream)
            _ = model_stage1(inputs)
            end.record(stream)

            torch.cuda.synchronize()
            t1[i] = start.elapsed_time(end)

    with torch.no_grad():
        for i in range(repetition):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record(stream)
            _ = model_stage2(1, inputs.shape[0], [inputs.shape[0]], inputs)
            end.record(stream)

            torch.cuda.synchronize()
            t2[i] = start.elapsed_time(end)

    return np.mean(t1), np.std(t1), np.mean(t2), np.std(t2)

def plot_confusion_matrix(confusion_matrix, groups, mv_type, output_path, save=False):
    plt.imshow(confusion_matrix, cmap='Reds')
    plt.colorbar()
    plt.ylabel('Ground Truths')
    plt.xlabel('Predictions')
    plt.title(mv_type)

    for _, group in enumerate(groups):
        start_point = int(group[0])
        end_point = int(group[-1]) + 1
        plt.plot((start_point - 0.5, start_point - 0.5), (start_point - 0.5, end_point - 0.5), linewidth=1, color='blue')
        plt.plot((start_point - 0.5, end_point - 0.5), (start_point - 0.5, start_point - 0.5), linewidth=1, color='blue')
        plt.plot((end_point - 0.5, end_point - 0.5), (start_point - 0.5, end_point - 0.5), linewidth=1, color='blue')
        plt.plot((start_point - 0.5, end_point - 0.5), (end_point - 0.5, end_point - 0.5), linewidth=1, color='blue')

    if save:
        plt.savefig(output_path + mv_type + '.pdf', dpi=1200, format='pdf', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

class CosineDecayLR:
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        super().__init__()

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

