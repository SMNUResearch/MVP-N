# Define dataset
DATA_ROOT = "./data/"
MAX_NUM_VIEWS = 6
IMAGE_SIZE = 224
HPIQ = False
DATA_INFO = {"CLASSES":['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
                        '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021',
                        '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', '0031', '0032',
                        '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040', '0041', '0042', '0043'],
             "GROUPS":[['0000', '0001'], ['0002', '0003'], ['0004', '0005'], ['0006', '0007'], ['0008', '0009'],
                      ['0010', '0011', '0012', '0013', '0014'], ['0015', '0016'], ['0017', '0018'], ['0019', '0020'],
                      ['0021', '0022'], ['0023', '0024'], ['0025', '0026'], ['0027', '0028', '0029', '0030'], ['0031', '0032'],
                      ['0033', '0034'], ['0035', '0036'], ['0037', '0038'], ['0039', '0040'], ['0041', '0042', '0043']],
             "NUM_CLASSES":44,
             "NUM_GROUPS":19}

# Define seed
SEED = 1000 # options: [100, 1000, 10000, 100000, 1000000]

# Define device
DEVICE = "cuda:0"

# Define backbone
ARCHITECTURE = "RESNET18"
FEATURE_DIM = 512

# Define train/test hyper-parameters
NUM_WORKERS = 2

SV_EPOCHS = 30
TRAIN_SV_BS = 128
TEST_SV_BS = 1
SV_MOMENTUM = 0.9
SV_WEIGHT_DECAY = 1e-3
SV_LR_INIT = 1e-2

MV_EPOCHS = 50
MV_WARMUP_EPOCHS = 1
TRAIN_MV_BS = 32
TEST_MV_BS = 1
MV_MOMENTUM = 0.9
MV_WEIGHT_DECAY = 1e-3
MV_LR_INIT = 1e-3
MV_LR_END = 0

TEST_T = 1

# Define soft label hyper-parameters; F: Fixed
SV_TYPE = "CE" # options: [CE, KD, SB, HB, LS, DSB, DHB, SAT, LRT, PLC, SEAL, OLS, HPIQ]

KD_T = 2
KD_ALPHA = 0.9
BS_CHECKPOINT = 1
SB_BETA = 0.5
HB_BETA = 0.95
LS_SMOOTH = 0.1
DB_CHECKPOINT = 1
DB_ALPHAS_F = [1., 2.]
DB_BETAS_F = [2., 1.]
DB_LAMBDAS_F = [0.5, 0.5]
DB_MAX_LOSS_BOUND_F = 95
DB_MIN_LOSS_BOUND_F = 5
DB_LOSS_BOUND_F = 1e-3
DB_RESOLUTION_F = 1000
DB_MAX_ITERATION_F = 100
DB_AVOID_ZERO_EPS_F = 1e-12
DB_EM_EPS_F = 1e-4
DB_NAN_EPS_F = 1e-12
SAT_CHECKPOINT = 4
SAT_ALPHA = 0.9
LRT_RETRO_EPOCH = 2
LRT_UPDATE_EPOCH = 4
LRT_INTERVAL_EPOCH = 6
LRT_EVERY_N_EPOCH = 1
LRT_DELTA_X_F = 1.2
LRT_DELTA_Y_F = 0.02
LRT_SOFT_EPS_F = 1e-3
LRT_UPDATE_EPS_F = 1e-4
LRT_RHO_F = 0.9
LRT_FLIP_EPS_F = 1e-2
PLC_ROLL_WINDOW = 5
PLC_WARM_UP = 2
PLC_DELTA = 0.2
PLC_STEP_SIZE = 0.05
PLC_LRT_RATIO_F = 1e-3
PLC_MAX_DELTA_F = 0.9
SEAL_TIME = 2
OLS_ALPHA = 0.9

# Define multi-view feature aggregation hyper-parameters; F: Fixed
MV_TYPE = "MVCNN_NEW" # options: [MVCNN_NEW, GVCNN, DAN, CVR]

GVCNN_M = 6
DAN_H = 2
DAN_NUM_HEADS_F = 1
DAN_INNER_DIM_F = 2048
DAN_DROPOUT_F = 0.1
CVR_K = 3
CVR_LAMBDA = 1.0
CVR_NUM_HEADS_F = 8
CVR_INNER_DIM_F = 2048
CVR_NORM_EPS_F = 1e-6
CVR_OTK_HEADS_F = 1
CVR_OTK_EPS_F = 0.05
CVR_OTK_MAX_ITER_F = 100
CVR_DROPOUT_F = 0
CVR_COORD_DIM_F = 64

# Define computation
REPETITION = 500

# Define path
SV_WEIGHT_PATH = "./weight_sv/"
SV_OUTPUT_PATH = "./output_sv/"
SV_TEST_WEIGHT = "./weight_sv/CE.pt"
MV_WEIGHT_PATH = "./weight_mv/"
MV_TEST_WEIGHT = "./weight_mv/DAN.pt"

# Define flag
SV_FLAG = "TRAIN" # options: [TRAIN, TEST]
MV_FLAG = "TRAIN" # options: [TRAIN, TEST, CM, COMPUTATION]

