#!/bin/bash


lr=0.01
steps=10
#max_norm=0.01 
max_norm=0.03125
alpha=1.0
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
model=vgg
optim=sgldm
noise_std=0.001
noise_type=normal
model_out=./checkpoint/${data}_${model}_adv_${optim}_${noise_type}_${noise_std}
echo "model_out: " ${model_out}
python ./main_adv_sgld.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --noise-std ${noise_std} \
                        --noise-type ${noise_type} \
                        --model_out ${model_out}.pth \
                        --optim ${optim} \
                        #--resume \
                        
