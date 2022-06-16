import os
import sys
import torch
import random
import warnings
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from termcolor import cprint
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True
warnings.filterwarnings('ignore')

import utils as tool
import parser as parser
from model.svcnn import SVCNN
from model.mvcnn_new import MVCNNNew
from model.gvcnn import GVCNN
from model.dan import DAN
from model.cvr import CVR
from dataset_single_view import SingleViewDataset
from dataset_multi_view import MultiViewDataset
from loss import LabelCrossEntropy
from engine_single_view import SingleViewEngine
from engine_multi_view import MultiViewEngine

if __name__ == '__main__':
    # set options
    opt = parser.get_parser()
    cprint('*'*25 + ' Start ' + '*'*25, 'yellow')

    # set seed
    seed = opt.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    # set cudnn
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True

    # set device
    device = torch.device(opt.DEVICE if torch.cuda.is_available() else 'cpu')

    # define model
    model_stage1 = SVCNN(opt.NUM_CLASSES, opt.ARCHITECTURE, opt.FEATURE_DIM, pretrained=True).to(device)

    # define dataset
    train_dataset = SingleViewDataset(opt.CLASSES, opt.GROUPS, opt.NUM_CLASSES, opt.DATA_ROOT, 'train', HPIQ=opt.HPIQ)
    if opt.MV_FLAG == 'TRAIN':
        cprint('*'*15 + ' Stage 1 ' + '*'*15, 'yellow')
        print('Number of Training Images:', len(train_dataset))

    train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_SV_BS, num_workers=opt.NUM_WORKERS, shuffle=True, pin_memory=True, worker_init_fn=tool.seed_worker)

    # define optimizer
    optimizer = optim.SGD(model_stage1.parameters(), lr=opt.SV_LR_INIT, weight_decay=opt.SV_WEIGHT_DECAY, momentum=opt.SV_MOMENTUM)

    # define criterion
    criterion = LabelCrossEntropy()

    # define engine
    engine = SingleViewEngine(model_stage1, train_data, None, opt.NUM_CLASSES, optimizer, None, criterion, None, None, device, single_view=False)
 
    # run single view
    if opt.MV_FLAG == 'TRAIN':
        engine.train_base(opt.SV_EPOCHS, len(train_dataset))

    if opt.MV_FLAG == 'TRAIN':
        cprint('*'*15 + ' Stage 2 ' + '*'*15, 'yellow')

    # define model
    if opt.MV_TYPE == 'MVCNN_NEW':
        model_stage2 = MVCNNNew(model_stage1).to(device)
    elif opt.MV_TYPE == 'GVCNN':
        model_stage2 = GVCNN(model_stage1, opt.GVCNN_M, opt.ARCHITECTURE, opt.IMAGE_SIZE).to(device)
    elif opt.MV_TYPE == 'DAN':
        model_stage2 = DAN(model_stage1, opt.DAN_H, opt.FEATURE_DIM, opt.DAN_NUM_HEADS_F, opt.DAN_INNER_DIM_F, opt.DAN_DROPOUT_F).to(device)
    elif opt.MV_TYPE == 'CVR':
        model_stage2 = CVR(model_stage1, opt.CVR_K, opt.FEATURE_DIM, opt.CVR_NUM_HEADS_F, opt.CVR_INNER_DIM_F, opt.CVR_NORM_EPS_F,
                           opt.CVR_OTK_HEADS_F, opt.CVR_OTK_EPS_F, opt.CVR_OTK_MAX_ITER_F, opt.CVR_DROPOUT_F, opt.CVR_COORD_DIM_F).to(device)

    if opt.MV_FLAG in ['TEST', 'CM']:
        model_stage2.load_state_dict(torch.load(opt.MV_TEST_WEIGHT, map_location=device))
        model_stage2.eval()

    # define dataset
    train_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'train', opt.MAX_NUM_VIEWS)
    valid_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'valid', opt.MAX_NUM_VIEWS)
    test_dataset = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'test', opt.MAX_NUM_VIEWS)
    if opt.MV_FLAG in ['TRAIN', 'TEST']:
        print('Number of Training Sets:', len(train_dataset))
        print('Number of Valid Sets:', len(valid_dataset))
        print('Number of Test Sets:', len(test_dataset))

    train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=True, pin_memory=True, worker_init_fn=tool.seed_worker)
    valid_data = DataLoader(valid_dataset, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)
    test_data = DataLoader(test_dataset, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)

    # define optimizer
    optimizer = optim.SGD(model_stage2.parameters(), lr=opt.MV_LR_INIT, weight_decay=opt.MV_WEIGHT_DECAY, momentum=opt.MV_MOMENTUM)
    scheduler = tool.CosineDecayLR(optimizer, T_max=opt.MV_EPOCHS*len(train_data), lr_init=opt.MV_LR_INIT, 
                                   lr_min=opt.MV_LR_END, warmup=opt.MV_WARMUP_EPOCHS*len(train_data))

    # set path
    if opt.MV_FLAG == 'TRAIN':
        if not os.path.exists(opt.MV_WEIGHT_PATH):
            os.mkdir(opt.MV_WEIGHT_PATH)

    # define engine
    engine = MultiViewEngine(model_stage2, train_data, valid_data, opt.NUM_CLASSES, optimizer, scheduler, criterion, opt.MV_WEIGHT_PATH, device, opt.MV_TYPE)

    # run multi-view
    if opt.MV_FLAG == 'TRAIN':
        if opt.MV_TYPE in ['MVCNN_NEW', 'GVCNN', 'DAN']:
            engine.train_base(opt.MV_EPOCHS)
        elif opt.MV_TYPE == 'CVR':
            vert = tool.get_vert(opt.CVR_K)
            engine.train_cvr(opt.MV_EPOCHS, vert, opt.CVR_LAMBDA, opt.CVR_NORM_EPS_F)
    elif opt.MV_FLAG == 'TEST':
        cprint('*'*10 + ' Valid Sets ' + '*'*10, 'yellow')
        engine.test(valid_data, opt.TEST_T)
        cprint('*'*10 + ' Test Sets ' + '*'*10, 'yellow')
        engine.test(test_data, opt.TEST_T)
    elif opt.MV_FLAG == 'COMPUTATION':
        cprint('*'*10 + ' Computational Efficiency ' + '*'*10, 'yellow')
        # define inputs
        inputs = torch.randn(opt.MAX_NUM_VIEWS, 3, opt.IMAGE_SIZE, opt.IMAGE_SIZE).to(device)
        # measure parameters
        p1, p2 = tool.get_parameters(model_stage1, model_stage2)
        # measure FLOPs
        f1, f2 = tool.get_FLOPs(model_stage1, model_stage2, inputs)
        # measure latency
        t1_mean, t1_std, t2_mean, t2_std = tool.get_time(model_stage1, model_stage2, inputs, opt.REPETITION)
        cprint('*'*10 + ' SVCNN ' + '*'*10, 'yellow')
        print('Model Size (M):', '%.2f' % (p1 / 1e6))
        print('FLOPs (G):', '%.2f' % (f1 / 1e9))
        print('Latency (ms):', '%.2f' % (t1_mean) + ' (mean)', '%.2f' % (t1_std) + ' (std)')
        cprint('*'*10 + ' ' + opt.MV_TYPE + ' ' + '*'*10, 'yellow')
        print('Model Size (M):', '%.2f' % (p2 / 1e6))
        print('FLOPs (G):', '%.2f' % (f2 / 1e9))
        print('Latency (ms):', '%.2f' % (t2_mean) + ' (mean)', '%.2f' % (t2_std) + ' (std)')
    elif opt.MV_FLAG == 'CM':
        cprint('*'*10 + ' Confusion Matrix ' + '*'*10, 'yellow')
        confusion_matrix = engine.confusion_matrix(valid_data)
        tool.plot_confusion_matrix(confusion_matrix, opt.GROUPS, opt.MV_TYPE)

    cprint('*'*25 + ' Finish ' + '*'*25, 'yellow')

