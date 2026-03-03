#!/bin/bash

#these are the encodings that have the 2500th, 5000th and 7000th
#indices in the sorted list of 10000 random encodings
encodings=(
    PQGLATRKEDCNIHFWYVMS
    EAKYNFMRHQPIWLDSVCGT
    TSQNGYIHWMAKFLVDEPRC
)

out_filenames=(
    2500
    5000
    7500
)

#running FCGR generation for each of the three encodings and saving the output in a csv file
for i in "${!encodings[@]}"
do
    Rscript --vanilla FCGR_gen.R \
    --encoding "${encodings[$i]}" \
    --scaling_factor 0.865 \
    --resolution 35 \
    --output_file "data/${out_filenames[$i]}.csv"\
    --input_filename data/deeploc_clean.csv
done
