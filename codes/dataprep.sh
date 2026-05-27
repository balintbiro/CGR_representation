#!/bin/bash

#dataprep
names=(
    deeploc
    ec
    pfam
)

temp_files=(
    deeploc.fasta
    ec.csv
    pfam.csv
)

#running dataprep for all the three datasets
for i in "${!names[@]}"
do
    python trial.py \
    --logfile "../results/logs/dataprep.log" \
    --dataset_name "${names[$i]}" \
    --tempfile "../data/${temp_files[$i]}" \
    --outfile "../data/${names[$i]}_clean.csv"
done
