#!/bin/bash

model=vgg
defense=adv
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
n_ensemble=15
#steps=( 3 4 6 9 13 18 26 38 55 78 112 162 234 336 483 695 1000 )
steps=(10)
attack=Linf
#max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
#max_norm=0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02
#max_norm=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0
#max_norm=0,0.01,0.02
max_norm=0,0.015,0.035,0.055,0.7
optim=sgldm
noise_std=0.001
#model_test=./checkpoint/cifar10/cifar10_${model}_${defense}.pth
model_test=./checkpoint/${data}_${model}_${defense}_${optim}_${noise_std}_0.01.pth
echo "Attacking" ${model_test}
for k in "${steps[@]}"
do
    echo "running" $k "steps"
    python acc_under_attack.py \
        --model $model \
        --defense $defense \
        --data $data \
        --root $root \
        --n_ensemble $n_ensemble \
        --steps $k \
        --max_norm $max_norm \
        --model_test $model_test \
        --attack $attack
done
