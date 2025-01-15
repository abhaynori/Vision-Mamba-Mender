# Vision Mamba Mender

<div align="center"><img width="200" src="logo.jpg"/></div>

For more information,
please visit https://vision-mamba-mender.github.io/.
## Requirements

- causal_conv1d: ```pip install causal-conv1d```
- mamba: ```pip install mamba-ssm```
- ```pip install -U openmim```
- ```mim install mmcv```
- ```pip install mmsegmentation```
- ```pip install mmengine```
- ```pip install ftfy```

## Algorithms

### 🐍 Preparation

1. Train a pre-trained model:

```shell
bash scripts/train.sh
```

2. Select high-confidence samples and low-confidence samples：

```shell
bash scripts/sample_selection.sh
```

3. Extract intermediate features from the model, including activations and gradients of samples:

```shell
bash scripts/feature_selection.sh
```

### 🔍 Interpreting

4. Visualize external interactions of states or internal interactions of states

```shell
bash state_external_visualize.sh
bash state_internal_visualize.sh
```

5. Binarize external interactions of states or internal interactions of states

```shell
bash state_external_mask.sh
bash state_internal_mask.sh # Steps necessary for calibration!
```

6. Compute interpretability scores for external interactions of states or internal interactions of states

```shell
bash state_external_score.sh
bash state_internal_score.sh
```

### 🔧 Calibrating

7. Calibrate external interactions of states or internal interactions of states

```shell
bash scripts/train_repair.sh
```

# Future Plans

This repository contains the initial version of the code for the *Vision Mamba Mender* paper. Please stay tuned for further refinements and detailed explanations in future updates.

## Others

- The directory structure of the files outputted by the algorithm:

``` shell
output_path # Overall output directory as defined by you
    ├── exp_name # Experiment name defined by you
            ├── models
            ⎪       └── xxx.pth       
            ├── samples
            ⎪       ├── htrain # Selected high-confidence samples
            ⎪       └── ltrain # Selected low-confidence samples
            ├── features
            ⎪       ├── hdata # Selected intermediate features of the high-confidence samples
            ⎪       ⎪       └── xxx.pkl 
            ⎪       └── ldata # Selected intermediate features of the low-confidence samples
            ├── visualize
            ⎪       ├── hdata
            ⎪       ⎪       ├── external # Visualization results of external interactions of states
            ⎪       ⎪       ⎪       └── xxx.JPEG/PNG
            ⎪       ⎪       └── internal # Visualization results of internal interactions of states
            ⎪       └── ldata # Same as above
            ├── masks
            ⎪       ├── hdata
            ⎪       ⎪       ├── external # Binarization results of external interactions of states
            ⎪       ⎪       ⎪       └── xxx.pt
            ⎪       ⎪       └── internal # Binarization results of internal interactions of states
            ⎪       └── ldata # Same as above
            └── scores
                    ├── hdata
                    ⎪       ├── external # Interpretability scores of external interactions of states
                    ⎪       ⎪       └── xxx.PNG
                    ⎪       └── internal # Interpretability scores of internal interactions of states
                    └── ldata # Same as above
```
