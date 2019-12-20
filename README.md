# GM Final Project
Adversarial defense through adversarial training and Bayesian learning


## Data
+ CIFAR10

## Network
+ VGG16


## Defense methods
+ `Plain`: No defense
+ `RSE`: Random Self-ensemble
+ `Adv`: Adversarial training
+ `Adv_vi`: Adversarial training with VI
+ `Adv_sgld`: Adversarial training with SGLD

*Known bugs*: due to a known bug in PyTorch [#11742](https://github.com/pytorch/pytorch/issues/11742), we cannot run RSE/Adv-BNN with multi-GPUs.

## Howto
Run `bash train_{method}.sh` to run different adversarial attack methods. Hyperparameters are offered in the scripts. Modify the cifar10 root in the scripts before you run them. For example, run `bash train_adv_sgld.sh` to run adv. training with SGLD.


## Reference
[*Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network*]