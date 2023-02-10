#!/bin/bash

beta=0.75 #0.05
lam=1     #dtw 
dtw="False"
lr=0.001
val_itvl=500
val_size=50
num_epochs=15
batch_size=1
experiment=37
summary="Test1"
from_checkpoint="False"
permutation="False"
transform="fourier" #cosine
stft_small="True"  #True
ft_container="mag"
thet=1
mp_encoder="double"
mp_decoder="double"
mp_join="2D"
embed="stretch" #stretch

if [ -d "$HOME/PixInWav2/outputs/$experiment-$summary" ]; then
    while true; do
        read -p "Folder $experiment-$summary already exists. Do you wish to overwrite?" yn
        case $yn in
            [Yy]* ) rm -rf "$HOME/PixInWav2/outputs/$experiment-$summary"; break;;
            [Nn]* ) echo "Rename your experiment. Exiting... "; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi
mkdir -p "$HOME/PixInWav2/outputs/$experiment-$summary"

cat > "$HOME/PixInWav2/outputs/$experiment-$summary/parameters.txt" <<EOF
===Hyperparameters===:
lr: $lr
batch_size: $batch_size

===Loss func hyperparameters===:
beta: $beta
theta: $thet
DTW lambda: $lam (disregard if not using DTW)

===Architecture hyperparameters===:
Using permutation? $permutation
Transform: $transform
Using small bottleneck? $stft_small
What ft container? $ft_container
Embedding style: $embed
mp_encoder: $mp_encoder
mp_decoder: $mp_decoder
mp_join: $mp_join

===Training parameters===:
Epochs: $num_epochs

===Validation parameters===:
val_itvl: $val_itvl its
val_size: $val_size 

===Command to run===:
CUDA_VISIBLE_DEVICES=X,Y python3 $HOME/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed
EOF

CUDA_VISIBLE_DEVICES=6,7 python3 $HOME/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed

