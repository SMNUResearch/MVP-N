# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification (NeurIPS 2022)
*This is the official PyTorch implementation.* [[paper]](https://openreview.net/forum?id=HYELrdRdJI)
## Create issues from this repository
Please contact us at wangren@snu.ac.kr. We will reply the issue within 3 days.
## Update
- *2023.03.03* Correct the utilization estimation for DAN and CVR.
- *2023.02.25* Fix some bugs.
- *2023.02.22* Add the code for analyzing the utilization of informative views in the multi-view-based feature aggregation. Add SMVCNN [[weight]](https://drive.google.com/file/d/1-OzUCO9K_51wqCkL3Wdd3NXIv1HAYFmg/view?usp=share_link). Code optimization.
- *2023.02.15* Add HS [[weight]](https://drive.google.com/file/d/1JODvl0oC64aN2clTCErdQz4bnY4Av2wn/view?usp=share_link). Code optimization.
- *2023.01.25* Add MVFN [[weight]](https://drive.google.com/file/d/1tKUSXcMB5yNraFTm5bbx2__ygk9tbTIW/view?usp=share_link). Code optimization.
## Summary of 50 multi-view-based feature aggregation methods [[Details]](https://drive.google.com/file/d/1NryQBPcvdeOkwGXIsBlHqZ28J-fhrqPd/view?usp=share_link)  
Period: 2015.01 ~ 2022.12  
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
Step 1: Download pretrained weights from [[Google Drive]](https://drive.google.com/uc?export=download&id=1_SroOiy6Y-7OND93WLoa25q6GWjSSsTP)  
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
