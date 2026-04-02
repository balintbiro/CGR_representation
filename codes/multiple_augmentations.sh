conditions=(
    True
    False
)

for i in "${!conditions[@]}"
do
    python augmentation.py \
    --logfile "multiple_augs.log" \
    --datasets_dir "../data/spec_datasets" \
    --outfile "../results/multiple_augs_results.csv" \
    --mix "${conditions[$i]}" \
    --n 10
done