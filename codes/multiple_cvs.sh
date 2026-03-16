#!/bin/bash

#these are the encodings that have the 2500th, 5000th and 7000th
#indices in the sorted list of 10000 random encodings
names=(
    Min
    Q1
    Q2
    Q3
    Max
)

input_files=(
    min.csv
    q1.csv
    q2.csv
    q3.csv
    max.csv
)

#running cv for each of the five encodings and saving the outputs in a csv file
for i in "${!names[@]}"
do
    python cnn_cv.py \
    --logfile "multiple_cv.log" \
    --fcgr_matrix "../data/spec_datasets/${input_files[$i]}" \
    --outfile "../results/multiple_cv_results.csv" \
    --name "${names[$i]}" \
    --res 35 \
    --n 20
done
