# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification (NeurIPS 2022) [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/819b8452be7d6af1351d4c4f9cbdbd9b-Paper-Datasets_and_Benchmarks.pdf) [[Reviews]](https://openreview.net/forum?id=HYELrdRdJI)
*This is the official PyTorch implementation.*
## Create issues from this repository
Please contact us at wangren@snu.ac.kr. We will reply the issue within 3 days.
## Notice
- Related research work published in 2023 is summarized. One paper is added to the list below.
- Modifications on the summary file: The code of MVT (BMVC 2021) and iMHL (TIP 2018) is released. The code of View-GCN++ (TPAMI 2022) has yet to be released. MVT (BMVC 2021) satisfies P1 by analyzing its open-source implementation.
- Related multi-view-based feature aggregation methods for biomedical tasks will not be summarized here.
- New hypergraph-based methods will no longer be summarized here unless explicitly designed for multi-view object classification.
- There is a typo in the caption of Table 4 (NeurIPS 2022), which should be corrected as 'Backbone (ResNet-18): 11.20 M, **10.91 G**, and 6.19 ± 0.05 ms'.
## Summary of 51 multi-view-based feature aggregation methods [[Details]](https://drive.google.com/file/d/1Fm5LAgYxfP_2xqNYz8INy7J2Jz1JW3ZP/view?usp=sharing)
Period: 2015.01 ~ 2023.12  
Conferences: NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, AAAI, IJCAI, MM, WACV, BMVC, ACCV  
Journals: TPAMI, IJCV, TIP, TNNLS, TMM, TCSVT, TVCG, PR  
Workshops: NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV  
## Environment
Ubuntu 20.04.3 LTS  
Python 3.8.10  
CUDA 11.1  
cuDNN 8  
NVIDIA GeForce RTX 3090 GPU  
Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz  
## Setup
Step 1: Get repository  
```
git clone https://github.com/SMNUResearch/MVP-N.git
cd MVP-N
```
Step 2: Install dependencies  
```
sh install.sh
```
## Dataset Preparation
Step 1: Download data.zip from [[Google Drive]](https://drive.google.com/uc?export=download&id=1rbjFXLtXGYSsgFN2r9AZtWxOVHGF5jAS)  
Step 2: Place data.zip in this repository  
Step 3: Unzip data.zip  
```
unzip data.zip
```
## Quick Test
Step 1: Download pretrained weights from [[Google Drive]](https://drive.google.com/file/d/18VNrODK-cDxNpgYsm6DoYWvq7wSKM8pw/view?usp=sharing)  
Step 2: Place weights.zip in this repository  
Step 3: Unzip weights.zip  
```
unzip weights.zip -d weights
```
Step 4: Evaluation
```
# feature aggregation performance
python3 main_multi_view.py -MV_FLAG=TEST -MV_TYPE=DAN -MV_TEST_WEIGHT=./weights/DAN.pt

# confusion matrix
python3 main_multi_view.py -MV_FLAG=CM -MV_TYPE=DAN -MV_TEST_WEIGHT=./weights/DAN.pt

# computational efficiency
python3 main_multi_view.py -MV_FLAG=COMPUTATION -MV_TYPE=DAN

# soft label performance
python3 main_single_view.py -SV_FLAG=TEST -SV_TYPE=SAT -SV_TEST_WEIGHT=./weights/SAT.pt
```
## Training
Training with default configurations
```
# feature aggregation
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=DAN
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=CVR
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=SMVCNN -SMVCNN_USE_EMBED

# soft label
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=HPIQ
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=KD
```
Training with other configurations
```
# feature aggregation
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=CVR -CVR_LAMBDA=0.1 -CVR_K=2

# soft label
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=KD -KD_T=3
```
Details of configurations are provided in `config.py`
