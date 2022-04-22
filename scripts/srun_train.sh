#!/bin/bash

beta=0.75
lam=1
lr=0.001
val_itvl=500
val_size=50
num_epochs=2
batch_size=16
experiment=029
summary="2_LumaExp"
output="Outputs/$experiment-$summary.txt"
from_checkpoint="False"
permutation="False"
transform="fourier"
ft_container="mag"
thet=1
mp_encoder="double"
mp_decoder="double"
mp_join="2D"
embed="stretch"


# Develop
srun -u -w gpic07 --gres=gpu:2,gpumem:12G -p gpi.develop -o $output --time 1:59:59 --mem 50G python3 ~/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed

# Normal
# srun -u -w gpic13 --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --val_itvl $val_itvl --val_size $val_size --num_epochs $num_epochs --batch_size $batch_size --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join --embed $embed

# DataParallelFix
#srun -u --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav2/src/DataParallelFix.py --beta $beta --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --add_noise $add_noise --noise_kind $noise_kind --noise_amplitude $noise_amplitude --add_dtw_term $add_dtw_term --rgb $rgb --transform $transform --on_phase $on_phase --architecture $architecture
