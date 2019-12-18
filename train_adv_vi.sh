#!/bin/bash

lr=0.01
steps=10
#max_norm=0.01 
max_norm=0.03125
#sigma_0=0.08
sigma_0=0.05
#init_s=0.08
init_s=0.05
alpha=1.0
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
model=vgg
model_out=./checkpoint/${data}_${model}_adv_vi
echo "model_out: " ${model_out}
python ./main_adv_vi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #--resume \
                        
