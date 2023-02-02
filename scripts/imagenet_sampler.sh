#!/bin/bash

# Makes a new directory with the reduced dataset
# Samples N train images of each class

# Test and validation images are omitted
# 'Annotation' and 'ImageSets' directories are ommited
# Sample submission and solutions are ommited

# Parameters
sample_size=15 # x1000 classes
dest_dir='sample' # Will mkdir
input_dir='.'
seed=2023



# Create folders
mkdir -p "$dest_dir/ILSVRC/Data/CLS-LOC/train"

# Copy mappings file
cp "$input_dir/LOC_synset_mapping.txt" "$dest_dir/mappings.txt"

get_seeded_random()
{
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

# Iterate all lines of the mappings file (all classes)
while read p; do
    class=$(echo "$p" | cut -f 1 -d " ")
    mkdir "$dest_dir/ILSVRC/Data/CLS-LOC/train/$class"

    # Get "sample_size" random images from that class
    mapfile -t imgs < <(ls "$input_dir/ILSVRC/Data/CLS-LOC/train/$class" | sort --random-source=<(get_seeded_random $seed) -R | head -n $sample_size)

    # Copy the images to the new directory
    for i in "${imgs[@]}"; do
        cp "$input_dir/ILSVRC/Data/CLS-LOC/train/$class/$i" "$dest_dir/ILSVRC/Data/CLS-LOC/train/$class/$i"
    done
done <"$dest_dir/mappings.txt"
