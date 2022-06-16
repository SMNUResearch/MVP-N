# MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification
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

# soft label
python3 main_single_view.py -SV_FLAG=TRAIN -SV_TYPE=HPIQ -HPIQ=True
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
