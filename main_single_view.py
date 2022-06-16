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
from dataset_single_view import SingleViewDataset
from dataset_multi_view import MultiViewDataset
from loss import LabelCrossEntropy, KDLoss, SoftBootstrapping, HardBootstrapping, LabelSmoothing
from loss import BetaMixture, DynamicBootstrapping, LikelihoodRatioTest, SelfAdaptiveTraining
from loss import ProgressiveLabelCorrection, OnlineLabelSmoothing
from engine_single_view import SingleViewEngine

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
    if opt.SV_FLAG in ['TEST']:
        model_stage1.load_state_dict(torch.load(opt.SV_TEST_WEIGHT, map_location=device))
        model_stage1.eval()

    # define dataset
    train_dataset = SingleViewDataset(opt.CLASSES, opt.GROUPS, opt.NUM_CLASSES, opt.DATA_ROOT, 'train', HPIQ=opt.HPIQ)
    valid_dataset = SingleViewDataset(opt.CLASSES, opt.GROUPS, opt.NUM_CLASSES, opt.DATA_ROOT, 'valid', HPIQ=opt.HPIQ)
    test_dataset = SingleViewDataset(opt.CLASSES, opt.GROUPS, opt.NUM_CLASSES, opt.DATA_ROOT, 'test', HPIQ=opt.HPIQ)
    print('Number of Training Images:', len(train_dataset))
    print('Number of Valid Images:', len(valid_dataset))
    print('Number of Test Images:', len(test_dataset))
    train_data = DataLoader(train_dataset, batch_size=opt.TRAIN_SV_BS, num_workers=opt.NUM_WORKERS, shuffle=True, pin_memory=True, worker_init_fn=tool.seed_worker)
    valid_data = DataLoader(valid_dataset, batch_size=opt.TEST_SV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)
    test_data = DataLoader(test_dataset, batch_size=opt.TEST_SV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)

    # define optimizer
    optimizer = optim.SGD(model_stage1.parameters(), lr=opt.SV_LR_INIT, weight_decay=opt.SV_WEIGHT_DECAY, momentum=opt.SV_MOMENTUM)

    # define criterion
    if opt.SV_TYPE in ['CE', 'SEAL', 'HPIQ']:
        criterion = LabelCrossEntropy()
    elif opt.SV_TYPE == 'KD':
        criterion = LabelCrossEntropy()
        criterion_student = KDLoss(opt.KD_T, opt.KD_ALPHA)
    elif opt.SV_TYPE == 'SB':
        criterion = SoftBootstrapping(opt.SB_BETA, opt.BS_CHECKPOINT)
    elif opt.SV_TYPE == 'HB':
        criterion = HardBootstrapping(opt.HB_BETA, opt.BS_CHECKPOINT)
    elif opt.SV_TYPE == 'LS':
        criterion = LabelSmoothing(opt.LS_SMOOTH)
    elif opt.SV_TYPE in ['DSB', 'DHB']:
        beta_model = BetaMixture(opt.DB_ALPHAS_F, opt.DB_BETAS_F, opt.DB_LAMBDAS_F, opt.DB_MAX_LOSS_BOUND_F, opt.DB_MIN_LOSS_BOUND_F, opt.DB_LOSS_BOUND_F, 
                                 opt.DB_RESOLUTION_F, opt.DB_MAX_ITERATION_F, opt.DB_AVOID_ZERO_EPS_F, opt.DB_EM_EPS_F, opt.DB_NAN_EPS_F)
        criterion = DynamicBootstrapping(beta_model, opt.DB_LOSS_BOUND_F, opt.DB_CHECKPOINT, opt.SV_TYPE, device)
    elif opt.SV_TYPE == 'SAT':
        criterion = SelfAdaptiveTraining(len(train_dataset), opt.NUM_CLASSES, device, opt.SAT_CHECKPOINT, opt.SAT_ALPHA)
    elif opt.SV_TYPE == 'LRT':
        criterion = LikelihoodRatioTest(len(train_dataset), opt.NUM_CLASSES, device, opt.LRT_DELTA_X_F, opt.LRT_DELTA_Y_F, opt.LRT_RETRO_EPOCH, opt.LRT_UPDATE_EPOCH, 
                                        opt.LRT_INTERVAL_EPOCH, opt.LRT_EVERY_N_EPOCH, opt.LRT_SOFT_EPS_F, opt.LRT_UPDATE_EPS_F, opt.LRT_RHO_F, opt.LRT_FLIP_EPS_F)
    elif opt.SV_TYPE == 'PLC':
        criterion = ProgressiveLabelCorrection(len(train_dataset), opt.NUM_CLASSES, device, opt.PLC_ROLL_WINDOW, opt.PLC_WARM_UP, 
                                               opt.PLC_DELTA, opt.PLC_STEP_SIZE, opt.PLC_LRT_RATIO_F, opt.PLC_MAX_DELTA_F)
    elif opt.SV_TYPE == 'OLS':
        criterion = OnlineLabelSmoothing(opt.NUM_CLASSES, device, opt.OLS_ALPHA)

    # set path
    if opt.SV_FLAG == 'TRAIN':
        if not os.path.exists(opt.SV_WEIGHT_PATH):
            os.mkdir(opt.SV_WEIGHT_PATH)

    if opt.SV_FLAG == 'TRAIN' and opt.SV_TYPE in ['KD', 'SEAL']:
        if not os.path.exists(opt.SV_OUTPUT_PATH):
            os.mkdir(opt.SV_OUTPUT_PATH)

    # define engine
    engine = SingleViewEngine(model_stage1, train_data, valid_data, opt.NUM_CLASSES, optimizer, opt.SV_TYPE, criterion, opt.SV_WEIGHT_PATH, opt.SV_OUTPUT_PATH, device, single_view=True)

    # run
    if opt.SV_FLAG == 'TRAIN':
        base_type = ['CE', 'LS', 'HPIQ']
        if opt.SV_TYPE in base_type:
            engine.train_base(opt.SV_EPOCHS, len(train_dataset))
        elif opt.SV_TYPE in ['SB', 'HB']:
            engine.train_bs(opt.SV_EPOCHS)
        elif opt.SV_TYPE == 'KD':
            cprint('*'*10 + ' Train Teacher Model ' + '*'*10, 'yellow')
            engine.train_base(opt.SV_EPOCHS, len(train_dataset), save_outputs=True)
            new_model_stage1 = SVCNN(opt.NUM_CLASSES, opt.ARCHITECTURE, opt.FEATURE_DIM, pretrained=True).to(device)
            new_optimizer = optim.SGD(new_model_stage1.parameters(), lr=opt.SV_LR_INIT, weight_decay=opt.SV_WEIGHT_DECAY, momentum=opt.SV_MOMENTUM)
            engine_student = SingleViewEngine(new_model_stage1, train_data, valid_data, opt.NUM_CLASSES, new_optimizer, opt.SV_TYPE, criterion_student, opt.SV_WEIGHT_PATH, opt.SV_OUTPUT_PATH, device, single_view=True)
            cprint('*'*10 + ' Train Student Model ' + '*'*10, 'yellow')
            engine_student.train_kd(opt.SV_EPOCHS, os.path.join(opt.SV_OUTPUT_PATH, opt.SV_TYPE + '.pt'))
        elif opt.SV_TYPE in ['DSB', 'DHB']:
            engine.train_db(opt.SV_EPOCHS)
        elif opt.SV_TYPE == 'SAT':
            engine.train_sat(opt.SV_EPOCHS)
        elif opt.SV_TYPE in ['LRT', 'PLC']:
            engine.train_lrt_plc(opt.SV_EPOCHS)
        elif opt.SV_TYPE == 'SEAL':
            for i in range(0, opt.SEAL_TIME + 1):
                if i == 0: # base training
                    cprint('*'*10 + ' Train Teacher Model ' + '*'*10, 'yellow')
                    engine.train_seal(opt.SV_EPOCHS, len(train_dataset), i, seal_targets=None)
                else:
                    cprint('*'*10 + ' Train Student Model ' + '*'*10, 'yellow')
                    new_model_stage1 = SVCNN(opt.NUM_CLASSES, opt.ARCHITECTURE, opt.FEATURE_DIM, pretrained=True).to(device)
                    new_optimizer = optim.SGD(new_model_stage1.parameters(), lr=opt.SV_LR_INIT, weight_decay=opt.SV_WEIGHT_DECAY, momentum=opt.SV_MOMENTUM)
                    seal_weight_path = opt.SV_WEIGHT_PATH + 'seal_' + str(i) + '/'
                    if not os.path.exists(seal_weight_path):
                        os.mkdir(seal_weight_path)

                    seal_targets = os.path.join(opt.SV_OUTPUT_PATH, opt.SV_TYPE + str(i - 1) + '.pt')
                    new_engine = SingleViewEngine(new_model_stage1, train_data, valid_data, opt.NUM_CLASSES, new_optimizer, opt.SV_TYPE, criterion, seal_weight_path, opt.SV_OUTPUT_PATH, device, single_view=True)
                    new_engine.train_seal(opt.SV_EPOCHS, len(train_dataset), i, seal_targets=seal_targets)
        elif opt.SV_TYPE == 'OLS':
            engine.train_ols(opt.SV_EPOCHS)
    elif opt.SV_FLAG == 'TEST':
        valid_dataset_mv = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'valid', opt.MAX_NUM_VIEWS)
        test_dataset_mv = MultiViewDataset(opt.CLASSES, opt.NUM_CLASSES, opt.DATA_ROOT, 'test', opt.MAX_NUM_VIEWS)
        valid_data_mv = DataLoader(valid_dataset_mv, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)
        test_data_mv = DataLoader(test_dataset_mv, batch_size=opt.TEST_MV_BS, num_workers=opt.NUM_WORKERS, shuffle=False, pin_memory=True, worker_init_fn=tool.seed_worker)
        cprint('*'*10 + ' Valid Sets ' + '*'*10, 'yellow')
        engine.test(valid_data, opt.TEST_T)
        engine.score_fusion(valid_data_mv, opt.TEST_T)
        cprint('*'*10 + ' Test Sets ' + '*'*10, 'yellow')
        engine.test(test_data, opt.TEST_T)
        engine.score_fusion(test_data_mv, opt.TEST_T)

    cprint('*'*25 + ' Finish ' + '*'*25, 'yellow')

