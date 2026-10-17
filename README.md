Random Erasing: Classification Reproduction
This project reproduces selected experiments from:

Zhong et al., “Random Erasing Data Augmentation,” AAAI 2020.

Original repository: https://github.com/zhunzhong07/Random-Erasing

The original code is credited to its respective authors. The original LICENSE is retained.

Experiments
ResNet-20, ResNet-32, ResNet-44, ResNet-56 and ResNet-110 on CIFAR-10, CIFAR-100 and Fashion-MNIST.
PreAct-ResNet-18 on CIFAR-10.
Four filling variants: random values, ImageNet mean values, black filling and white filling.
PreAct-ResNet-18 evaluation under artificial occlusion from 0% to 50%, in steps of 5%.
Classification experiments used 300 epochs and five runs per condition, with seeds 42–46. Baseline models used p=0; RE models used p=0.5. Results are reported as mean test error ± sample standard deviation.

My modifications
Compatibility updates for the PyTorch environment used.
Random pixel filling in the classification transforms.
PreAct-ResNet-18 integration.
Separate scripts for the filling variants.
Occlusion evaluation and visual-check scripts.
ResNet experiments used independent random integers from 0–255 for each erased pixel/channel, divided by 255. PreAct-ResNet-18 used independent continuous random values between 0 and 1. Erasing was applied before normalization.

The occlusion evaluation uses independently sampled values for each pixel/channel. Its exact correspondence to the paper’s evaluation is unconfirmed, and the numerical difference is discussed in the report.

Files and results
The repository contains the original code, modified scripts, available run scripts, experiment logs and result CSVs.

The Table1_Random_Pixel folder organises the corresponding results by dataset, model, seed and baseline/RE condition.

Datasets and trained checkpoint weights are excluded from this upload. Checkpoint weights are retained locally. Occlusion evaluation requires the corresponding trained baseline and RE checkpoints.

## Classification Results

Test error (%), reported as mean ± sample standard deviation over
five runs using seeds 42–46. Lower is better.

| Model | CIFAR-10 Baseline | CIFAR-10 RE | CIFAR-100 Baseline | CIFAR-100 RE | Fashion-MNIST Baseline | Fashion-MNIST RE |
|---|---|---|---|---|---|---|
| ResNet-20 | 7.11 ± 0.06 | 6.75 ± 0.23 | 30.64 ± 0.25 | 29.75 ± 0.22 | 4.81 ± 0.18 | 4.52 ± 0.10 |
| ResNet-32 | 6.33 ± 0.23 | 5.83 ± 0.18 | 28.75 ± 0.34 | 26.99 ± 0.22 | 4.78 ± 0.12 | 4.61 ± 0.09 |
| ResNet-44 | 5.61 ± 0.27 | 5.11 ± 0.25 | 25.82 ± 0.76 | 24.03 ± 0.17 | 4.83 ± 0.11 | 4.69 ± 0.31 |
| ResNet-56 | 5.25 ± 0.19 | 4.98 ± 0.25 | 25.30 ± 0.44 | 23.79 ± 0.34 | 4.81 ± 0.16 | 4.56 ± 0.18 |
| ResNet-110 | 5.33 ± 0.14 | 4.54 ± 0.28 | 23.17 ± 0.64 | 22.02 ± 0.57 | 4.94 ± 0.06 | 4.58 ± 0.04 |
| PreAct-ResNet-18 | 5.39 ± 0.14 | 4.43 ± 0.07 | — | — | — | — |

Baseline: p = 0. RE: p = 0.5. Each run used 300 training epochs.

Person re-identification
The separate person_reid folder contains the adapted person re-identification implementation and its own source acknowledgements.
