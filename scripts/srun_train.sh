#!/bin/bash

beta=0.75
lam=0
lr=0.001
experiment=012
summary="2_TestPerm2"
output="Outputs/$experiment-$summary.txt"
from_checkpoint="False"
permutation="True"
transform="fourier"
ft_container="mag"
thet=1
mp_encoder="double"
mp_decoder="double"
mp_join="3D"


# Develop
# srun -u -w gpic07 --gres=gpu:2,gpumem:12G -p gpi.develop -o $output --time 1:59:59 --mem 50G python3 ~/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join

# Normal
srun -u -w gpic08 --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav2/src/main.py --beta $beta --lam $lam --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --permutation $permutation --transform $transform --ft_container $ft_container --thet $thet --mp_encoder $mp_encoder --mp_decoder $mp_decoder --mp_join $mp_join

# DataParallelFix
#srun -u --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav2/src/DataParallelFix.py --beta $beta --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --add_noise $add_noise --noise_kind $noise_kind --noise_amplitude $noise_amplitude --add_dtw_term $add_dtw_term --rgb $rgb --transform $transform --on_phase $on_phase --architecture $architecture
