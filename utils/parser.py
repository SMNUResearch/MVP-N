import yaml
import argparse

def get_parser(config_file):
    parser = argparse.ArgumentParser()
    config = yaml.load(open(config_file), Loader=yaml.SafeLoader)

    parser.add_argument('-DATA_ROOT', type=str, default=config['DATA_INFO']['DATA_ROOT'])
    parser.add_argument('-MAX_NUM_VIEWS', type=int, default=config['DATA_INFO']['MAX_NUM_VIEWS'])
    parser.add_argument('-IMAGE_SIZE', type=int, default=config['DATA_INFO']['IMAGE_SIZE'])
    parser.add_argument('-NUM_CLASSES', type=int, default=config['DATA_INFO']['NUM_CLASSES'])
    parser.add_argument('-CLASSES', type=str, default=config['DATA_INFO']['CLASSES'])
    parser.add_argument('-NUM_GROUPS', type=int, default=config['DATA_INFO']['NUM_GROUPS'])
    parser.add_argument('-GROUPS', type=str, default=config['DATA_INFO']['GROUPS'])

    parser.add_argument('-ARCHITECTURE', type=str, default=config['BACKBONE_INFO']['ARCHITECTURE'])
    parser.add_argument('-FEATURE_DIM', type=int, default=config['BACKBONE_INFO']['FEATURE_DIM'])

    parser.add_argument('-SEED', type=int, default=config['ENV_INFO']['SEED'])
    parser.add_argument('-DEVICE', type=str, default=config['ENV_INFO']['DEVICE'])
    parser.add_argument('-NUM_WORKERS', type=int, default=config['ENV_INFO']['NUM_WORKERS'])

    parser.add_argument('-SV_TYPE', type=str, default=config['SV_INFO']['SV_TYPE'])
    parser.add_argument('-SV_FLAG', type=str, default=config['SV_INFO']['SV_FLAG'])
    parser.add_argument('-SV_WEIGHT_PATH', type=str, default=config['SV_INFO']['SV_WEIGHT_PATH'])
    parser.add_argument('-SV_OUTPUT_PATH', type=str, default=config['SV_INFO']['SV_OUTPUT_PATH'])
    parser.add_argument('-SV_TEST_WEIGHT', type=str, default=config['SV_INFO']['SV_TEST_WEIGHT'])
    parser.add_argument('-SV_EPOCHS', type=int, default=config['SV_INFO']['SV_EPOCHS'])
    parser.add_argument('-SV_BS_TRAIN', type=int, default=config['SV_INFO']['SV_BS_TRAIN'])
    parser.add_argument('-SV_BS_TEST', type=int, default=config['SV_INFO']['SV_BS_TEST'])
    parser.add_argument('-SV_MOMENTUM', type=float, default=config['SV_INFO']['SV_MOMENTUM'])
    parser.add_argument('-SV_WEIGHT_DECAY', type=float, default=config['SV_INFO']['SV_WEIGHT_DECAY'])
    parser.add_argument('-SV_LR_INIT', type=float, default=config['SV_INFO']['SV_LR_INIT'])

    parser.add_argument('-KD_T', type=float, default=config['SOFT_LABEL_INFO']['KD_T'])
    parser.add_argument('-KD_ALPHA', type=float, default=config['SOFT_LABEL_INFO']['KD_ALPHA'])
    parser.add_argument('-BS_CHECKPOINT', type=int, default=config['SOFT_LABEL_INFO']['BS_CHECKPOINT'])
    parser.add_argument('-SB_BETA', type=float, default=config['SOFT_LABEL_INFO']['SB_BETA'])
    parser.add_argument('-HB_BETA', type=float, default=config['SOFT_LABEL_INFO']['HB_BETA'])
    parser.add_argument('-LS_SMOOTH', type=float, default=config['SOFT_LABEL_INFO']['LS_SMOOTH'])
    parser.add_argument('-DB_CHECKPOINT', type=int, default=config['SOFT_LABEL_INFO']['DB_CHECKPOINT'])
    parser.add_argument('-DB_ALPHAS_F', type=str, default=config['SOFT_LABEL_INFO']['DB_ALPHAS_F'])
    parser.add_argument('-DB_BETAS_F', type=str, default=config['SOFT_LABEL_INFO']['DB_BETAS_F'])
    parser.add_argument('-DB_LAMBDAS_F', type=str, default=config['SOFT_LABEL_INFO']['DB_LAMBDAS_F'])
    parser.add_argument('-DB_MAX_LOSS_BOUND_F', type=int, default=config['SOFT_LABEL_INFO']['DB_MAX_LOSS_BOUND_F'])
    parser.add_argument('-DB_MIN_LOSS_BOUND_F', type=int, default=config['SOFT_LABEL_INFO']['DB_MIN_LOSS_BOUND_F'])
    parser.add_argument('-DB_LOSS_BOUND_F', type=float, default=config['SOFT_LABEL_INFO']['DB_LOSS_BOUND_F'])
    parser.add_argument('-DB_RESOLUTION_F', type=int, default=config['SOFT_LABEL_INFO']['DB_RESOLUTION_F'])
    parser.add_argument('-DB_MAX_ITERATION_F', type=int, default=config['SOFT_LABEL_INFO']['DB_MAX_ITERATION_F'])
    parser.add_argument('-DB_AVOID_ZERO_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['DB_AVOID_ZERO_EPS_F'])
    parser.add_argument('-DB_EM_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['DB_EM_EPS_F'])
    parser.add_argument('-DB_NAN_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['DB_NAN_EPS_F'])
    parser.add_argument('-SAT_CHECKPOINT', type=int, default=config['SOFT_LABEL_INFO']['SAT_CHECKPOINT'])
    parser.add_argument('-SAT_ALPHA', type=float, default=config['SOFT_LABEL_INFO']['SAT_ALPHA'])
    parser.add_argument('-LRT_RETRO_EPOCH', type=int, default=config['SOFT_LABEL_INFO']['LRT_RETRO_EPOCH'])
    parser.add_argument('-LRT_UPDATE_EPOCH', type=int, default=config['SOFT_LABEL_INFO']['LRT_UPDATE_EPOCH'])
    parser.add_argument('-LRT_INTERVAL_EPOCH', type=int, default=config['SOFT_LABEL_INFO']['LRT_INTERVAL_EPOCH'])
    parser.add_argument('-LRT_EVERY_N_EPOCH', type=int, default=config['SOFT_LABEL_INFO']['LRT_EVERY_N_EPOCH'])
    parser.add_argument('-LRT_DELTA_X_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_DELTA_X_F'])
    parser.add_argument('-LRT_DELTA_Y_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_DELTA_Y_F'])
    parser.add_argument('-LRT_SOFT_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_SOFT_EPS_F'])
    parser.add_argument('-LRT_UPDATE_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_UPDATE_EPS_F'])
    parser.add_argument('-LRT_RHO_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_RHO_F'])
    parser.add_argument('-LRT_FLIP_EPS_F', type=float, default=config['SOFT_LABEL_INFO']['LRT_FLIP_EPS_F'])
    parser.add_argument('-PLC_ROLL_WINDOW', type=int, default=config['SOFT_LABEL_INFO']['PLC_ROLL_WINDOW'])
    parser.add_argument('-PLC_WARM_UP', type=int, default=config['SOFT_LABEL_INFO']['PLC_WARM_UP'])
    parser.add_argument('-PLC_DELTA', type=float, default=config['SOFT_LABEL_INFO']['PLC_DELTA'])
    parser.add_argument('-PLC_STEP_SIZE', type=float, default=config['SOFT_LABEL_INFO']['PLC_STEP_SIZE'])
    parser.add_argument('-PLC_LRT_RATIO_F', type=float, default=config['SOFT_LABEL_INFO']['PLC_LRT_RATIO_F'])
    parser.add_argument('-PLC_MAX_DELTA_F', type=float, default=config['SOFT_LABEL_INFO']['PLC_MAX_DELTA_F'])
    parser.add_argument('-SEAL_TIME', type=int, default=config['SOFT_LABEL_INFO']['SEAL_TIME'])
    parser.add_argument('-OLS_ALPHA', type=float, default=config['SOFT_LABEL_INFO']['OLS_ALPHA'])

    parser.add_argument('-MV_TYPE', type=str, default=config['MV_INFO']['MV_TYPE'])
    parser.add_argument('-MV_FLAG', type=str, default=config['MV_INFO']['MV_FLAG'])
    parser.add_argument('-MV_WEIGHT_PATH', type=str, default=config['MV_INFO']['MV_WEIGHT_PATH'])
    parser.add_argument('-MV_OUTPUT_PATH', type=str, default=config['MV_INFO']['MV_OUTPUT_PATH'])
    parser.add_argument('-MV_TEST_WEIGHT', type=str, default=config['MV_INFO']['MV_TEST_WEIGHT'])
    parser.add_argument('-MV_EPOCHS', type=int, default=config['MV_INFO']['MV_EPOCHS'])
    parser.add_argument('-MV_BS_TRAIN', type=int, default=config['MV_INFO']['MV_BS_TRAIN'])
    parser.add_argument('-MV_BS_TEST', type=int, default=config['MV_INFO']['MV_BS_TEST'])
    parser.add_argument('-MV_MOMENTUM', type=float, default=config['MV_INFO']['MV_MOMENTUM'])
    parser.add_argument('-MV_WEIGHT_DECAY', type=float, default=config['MV_INFO']['MV_WEIGHT_DECAY'])
    parser.add_argument('-MV_WARMUP_EPOCHS', type=int, default=config['MV_INFO']['MV_WARMUP_EPOCHS'])
    parser.add_argument('-MV_LR_INIT', type=float, default=config['MV_INFO']['MV_LR_INIT'])
    parser.add_argument('-MV_LR_END', type=float, default=config['MV_INFO']['MV_LR_END'])

    parser.add_argument('-GVCNN_M', type=int, default=config['FEATURE_AGGREGATION_INFO']['GVCNN_M'])
    parser.add_argument('-DAN_H', type=int, default=config['FEATURE_AGGREGATION_INFO']['DAN_H'])
    parser.add_argument('-DAN_NUM_HEADS_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['DAN_NUM_HEADS_F'])
    parser.add_argument('-DAN_INNER_DIM_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['DAN_INNER_DIM_F'])
    parser.add_argument('-DAN_DROPOUT_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['DAN_DROPOUT_F'])
    parser.add_argument('-CVR_K', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_K'])
    parser.add_argument('-CVR_LAMBDA', type=float, default=config['FEATURE_AGGREGATION_INFO']['CVR_LAMBDA'])
    parser.add_argument('-CVR_NUM_HEADS_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_NUM_HEADS_F'])
    parser.add_argument('-CVR_INNER_DIM_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_INNER_DIM_F'])
    parser.add_argument('-CVR_NORM_EPS_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['CVR_NORM_EPS_F'])
    parser.add_argument('-CVR_OTK_HEADS_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_OTK_HEADS_F'])
    parser.add_argument('-CVR_OTK_EPS_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['CVR_OTK_EPS_F'])
    parser.add_argument('-CVR_OTK_MAX_ITER_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_OTK_MAX_ITER_F'])
    parser.add_argument('-CVR_DROPOUT_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['CVR_DROPOUT_F'])
    parser.add_argument('-CVR_COORD_DIM_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['CVR_COORD_DIM_F'])
    parser.add_argument('-SMVCNN_D', type=int, default=config['FEATURE_AGGREGATION_INFO']['SMVCNN_D'])
    parser.add_argument('-SMVCNN_USE_EMBED', action='store_true')
    parser.add_argument('-VSF_NUM_LAYERS', type=int, default=config['FEATURE_AGGREGATION_INFO']['VSF_NUM_LAYERS'])
    parser.add_argument('-VSF_WIDENING_FACTOR', type=int, default=config['FEATURE_AGGREGATION_INFO']['VSF_WIDENING_FACTOR'])
    parser.add_argument('-VSF_NUM_HEADS_F', type=int, default=config['FEATURE_AGGREGATION_INFO']['VSF_NUM_HEADS_F'])
    parser.add_argument('-VSF_ATTENTION_DROPOUT_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['VSF_ATTENTION_DROPOUT_F'])
    parser.add_argument('-VSF_MLP_DROPOUT_F', type=float, default=config['FEATURE_AGGREGATION_INFO']['VSF_MLP_DROPOUT_F'])

    parser.add_argument('-TEST_T', type=float, default=config['TEST_T'])
    parser.add_argument('-REPETITION', type=int, default=config['REPETITION'])

    opt = parser.parse_args()

    return opt

