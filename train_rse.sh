#!/bin/bash

lr=0.01
noise_init=0.2
noise_inner=0.1
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
model=vgg
model_out=./checkpoint/${data}_${model}_rse
echo "model_out: " ${model_out}
python ./main_rse.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --noise_init ${noise_init} \
                        --noise_inner ${noise_inner} \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
