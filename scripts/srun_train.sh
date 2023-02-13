#!/bin/bash
source ini.sh

beta=0.05 #0.05
lam=1     #dtw 0.00001 else 1
dtw=false
lr=0.001
val_itvl=500
val_size=50
num_epochs=3
batch_size=1
experiment=1
summary="B4"
from_checkpoint=false
permutation=false
transform="cosine" #cosine
stft_small=true  #True
ft_container="mag"
thet=1
mp_encoder="double"
mp_decoder="double"
mp_join="2D"
embed="stretch" #stretch

if [ -d "$HOME/later2/PixInWav2/outputs/$experiment-$summary" ]; then
    while true; do
        read -p "Folder $experiment-$summary already exists. Do you wish to overwrite?" yn
        case $yn in
            [Yy]* ) rm -rf "$HOME/later2/PixInWav2/outputs/$experiment-$summary"; break;;
            [Nn]* ) echo "Rename your experiment. Exiting... "; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi
mkdir -p "$HOME/later2/PixInWav2/outputs/$experiment-$summary"

cat > "$HOME/later2/PixInWav2/outputs/$experiment-$summary/parameters.txt" <<EOF
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
CUDA_VISIBLE_DEVICES=X,Y python3 $HOME/later2/PixInWav2/src/main.py --beta $beta --lam $lam --dtw $dtw --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed
EOF

CUDA_VISIBLE_DEVICES=1 python3 $HOME/later2/PixInWav2/src/main.py --beta $beta --lam $lam --dtw $dtw --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --stft_small $stft_small --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed

