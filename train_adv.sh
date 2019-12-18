#!/bin/bash

lr=0.01
steps=10
max_norm=0.03125
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
model=vgg
model_out=./checkpoint/${data}_${model}_adv
echo "model_out: " ${model_out}
python ./main_adv.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
