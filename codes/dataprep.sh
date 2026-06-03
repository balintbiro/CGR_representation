#!/bin/bash

#dataprep
names=(
    deeploc
    tox
    pfam
    immune
)

temp_files=(
    deeploc.fasta
    tox.csv
    pfam.csv
    immune.csv
)

#running dataprep for all the three datasets
for i in "${!names[@]}"
do
    python dataprep.py \
    --logfile "../results/logs/dataprep.log" \
    --dataset_name "${names[$i]}" \
    --tempfile "../data/${temp_files[$i]}" \
    --outfile "../data/${names[$i]}_clean.csv"
done
