# CGR_representation
Check whether the method for placing vertices during Chaos Game Representation (CGR) has an impact on biological sequence classification perofrmance.

```shell

# running random search of different CGR encodings. This script calls FCGR_gen.R from inside
python ./cnn_random_encoding_search.py \
    --logfile cnn.log \
    --seqfile ../data/deeploc_clean.csv \
    --fcgrfile ../data/random_encoding_0865_35.csv \
    --outfile ../data/cnn_res_iter.csv \
    --sf 0.865 \
    --res 35 \
    --n 10_000
```
The resulting file is `results/cnn_res_iter.csv`.

```shell

# creating the dedicated encodings (Min, Q1...Max).
# It uses the FCGR generation script (FCGR_gen.R)
bash dedicated_encodings.sh
```

```shell

# running cross validation for the dedicated datasets.
# It calls internally the cnn_cv.py file
bash multiple_cvs.sh
```
The resulting file is `results/multiple_cv_results.csv`.

```shell

# running augmentation with cross validation.
# it calls the augmentation.py file
bash multiple_augmentations.sh
```
The resulting file is `results/multiple_aug_results.csv`.

# Benchmark datasets:
- Deeploc1.0
    - reference https://academic.oup.com/bioinformatics/article/33/21/3387/3931857
    - dataset https://services.healthtech.dtu.dk/services/DeepLoc-1.0/
