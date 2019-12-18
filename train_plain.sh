#!/bin/bash

lr=0.01
data=cifar10
root=/dvmm-filer2/datasets/CIFAR/
model=vgg
model_out=./checkpoint/${data}_${model}_plain
echo "model_out: " ${model_out}
python ./main_plain.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
