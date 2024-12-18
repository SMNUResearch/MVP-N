# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification (NeurIPS 2022) [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/819b8452be7d6af1351d4c4f9cbdbd9b-Paper-Datasets_and_Benchmarks.pdf) [[Reviews]](https://openreview.net/forum?id=HYELrdRdJI)
# Towards Real-World Multi-View Object Classification: Dataset, Benchmark, and Analysis (TCSVT 2024) [[Paper]](https://ieeexplore.ieee.org/document/10416228) [[Reviews]](https://drive.google.com/file/d/1bElWM0qzlEPEwzTiON5HSXjo1j9WUixG/view?usp=sharing)
*This is the official PyTorch implementation.*
## Create issues from this repository
Please contact us at wangren@snu.ac.kr. We will reply the issue within 3 days.
## Notice Board
- Related research work published in 2024.01 ~ 2024.06 is summarized. Five papers are added to the list below.
- The usage of FG3D dataset (TIP 2021) is supported in this repository.
- VSFormer (TVCG 2024) is added to this benchmark.
- Related research work published in 2023 is summarized. One paper is added to the list below.
- Related multi-view-based feature aggregation methods for biomedical tasks will not be summarized here.
- New hypergraph-based methods and soft label methods published from 2023 will no longer be summarized here unless explicitly designed for multi-view object classification.
## Clarification
- Discussion on the usage of `allow_tf32`.
  ```
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  ```
    - Except for CVR (ICCV 2021), the performance of other methods is almost unaffected.
    - The performance change for CVR cannot be neglected and may affect its best configuration.
    - We recommend trying the following configurations to get the best one when doing custom training on the MVP-N dataset.
    ```
    -CVR_K=2 -CVR_LAMBDA=1
    -CVR_K=3 -CVR_LAMBDA=0.5
    -CVR_K=3 -CVR_LAMBDA=1
    ```
- Modifications in the summary file compared to the paper content.
    - The codes of MVT (BMVC 2021) and iMHL (TIP 2018) are released.
    - MVT (BMVC 2021) satisfies P1 by analyzing its open-source implementation.
- There is a typo in the caption of Table 4 (NeurIPS 2022), which should be corrected as 'Backbone (ResNet-18): 11.20 M, **10.91 G**, and 6.19 ± 0.05 ms'.
- There is a typo in the second row of TABLE VII (TCSVT 2024), which should be corrected as '**97.97**'.
## Summary of 56 multi-view-based feature aggregation methods [[Details]](https://drive.google.com/file/d/1vhcg9w-PcUoqTtvd-a608orHqSnYxeVb/view?usp=drive_link)
Period: 2015.01 ~ 2024.06  
Conferences: NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, AAAI, IJCAI, MM, WACV, BMVC, ACCV  
Journals: TPAMI, IJCV, TIP, TNNLS, TMM, TCSVT, TVCG, PR  
Workshops: NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV  
| Year | Conferences | Journals | Workshops |
|  :--------  |  :-------:  |  :-------:  |  :-------:  |
| 2015 | 1 | 0 | 0 |
| 2016 | 2 | 0 | 0 |
| 2017 | 2 | 1 | 0 |
| 2018 | 5 | 5 | 1 |
| 2019 | 7 | 4 | 0 |
| 2020 | 4 | 2 | 0 |
| 2021 | 4 | 6 | 0 |
| 2022 | 2 | 5 | 2 |
| 2023 | 0 | 1 | 0 |
| 2024 | 1 | 4 | 0 |
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
Step 1: Get dataset documentation from [[Google Drive]](https://drive.google.com/file/d/167Om0A5rl7s3yxQLULbbJ7KkRXcgVHbC/view?usp=sharing)  
Step 2: Download data.zip from [[Google Drive]](https://drive.google.com/uc?export=download&id=1rbjFXLtXGYSsgFN2r9AZtWxOVHGF5jAS)  
Step 3: Place data.zip in this repository  
Step 4: Unzip data.zip  
```
unzip data.zip
```
## Quick Test
Step 1: Download pretrained weights from [[Google Drive]](https://drive.google.com/file/d/1W1GuSrD2Pb4k292Ag1ntrlm_DtojfA3Y/view?usp=sharing)  
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
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=KD
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=HPIQ
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=HS
```
Training with other configurations
```
# feature aggregation
python3 main_multi_view.py -MV_FLAG=TRAIN -MV_TYPE=CVR -CVR_LAMBDA=0.5 -CVR_K=3

# soft label
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=KD -KD_T=3
```
Details of configurations are provided in `config/base.yaml`
## Training (FG3D)
Step 1: Download FG3D.zip from [[Google Drive]](https://drive.google.com/file/d/1MY6wJldAglCdJr3m7PBQ53PpuNST2nu4/view?usp=sharing)  
Step 2: Place FG3D.zip in this repository  
Step 3: Unzip FG3D.zip  
```
unzip FG3D.zip
```
Step 4: Training with default configurations. Details are provided in `config/FG3D.yaml`
```
python3 main_multi_view_FG3D.py -MV_FLAG=TRAIN -MV_TYPE=DAN -NUM_CLASSES=13 -CLASSES=Airplane
python3 main_multi_view_FG3D.py -MV_FLAG=TRAIN -MV_TYPE=SMVCNN -NUM_CLASSES=20 -CLASSES=Car
python3 main_multi_view_FG3D.py -MV_FLAG=TRAIN -MV_TYPE=VSF -NUM_CLASSES=33 -CLASSES=Chair
```
