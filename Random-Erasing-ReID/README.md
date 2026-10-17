# Person Re-identification Reproduction

This folder contains code adapted from:
https://github.com/layumi/Person_reID_baseline_pytorch

The original code belongs to its respective authors. The original
LICENSE is retained.

## My modifications

The files `train_rr.py`, `test_rr.py`, `model_rr.py` and
`random_erasing_rr.py` contain my adaptations for the Random Erasing
reproduction experiments.

I evaluated IDE with ImageNet-pretrained ResNet-18, ResNet-34 and
ResNet-50 on Market-1501. Each backbone was trained for 60 epochs,
once without Random Erasing and once with Random Erasing.
Results were evaluated using Rank-1 accuracy and mAP.

The backbone was changed manually between experiments. The current
`model_rr.py` should be checked before running a particular backbone.

## Results and checkpoints

Experiment logs and configuration files are included in this folder name: checkpoint_from_re-ID.
Datasets and trained checkpoint weights are excluded because of their
size. The trained weights are retained locally.