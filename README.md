# CGR_representation
Check whether the method for placing vertices during Chaos Game Representation (CGR) has an impact on biological sequence classification perofrmance.

### Datapreparation

```shell
# running datapreparation (get and cleaning) for DeepLoc, EC and PFAM.
python -m codes.datapreparation.dataprep \
    --logfile dp.log \
    --dataset_name {dataset_name} \
    --tempfile {dataset_name}.csv \
    --outfile {dataset_name}_clean.csv
```
It produces:
- `dp.log`
- `{dataset_name}_clean.csv` with `sequence` and `label` columns

```shell
# running dedicated encoding generation
python -m codes.datapreparation.dedicated_encodings \
    --logfile dedicated_encodings.log \
    --model custom \
    --dataset_name pfam \
    --outdir dedicated_encodings/
```
It produces:
- `dedicated_encodings.log`
- `dedicated_encodings/pfam/custom/{min,q1,q2,q3,max}.csv`

```shell

# running random search of different CGR encodings. This script calls FCGR_gen.R from inside
python -m codes.cnn_random_encoding_search.py \
    --logfile results/logs/cnn.log \
    --seqfile data/{dataset_name}_clean.csv \
    --outfile results/cnn_res_iter.csv \
    --task binary \
    --dataset_name {dataset_name} \
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

### Benchmark datasets:
- Deeploc1.0
    - reference: https://academic.oup.com/bioinformatics/article/33/21/3387/3931857
    - dataset https://services.healthtech.dtu.dk/services/DeepLoc-1.0/
- Immune
    - references:
        - https://proceedings.iclr.cc/paper_files/paper/2025/hash/a9e8e05221b60d4161a26a00a8fd6c78-Abstract-Conference.html
    - dataset https://huggingface.co/datasets/AI4Protein/VenusVaccine_VirusBinary_ESMFold
- PFAM
    - references:
        - https://zenodo.org/records/8167436
        - https://academic.oup.com/nar/article/49/D1/D412/5943818
    - dataset https://zenodo.org/records/8167436/files/pfam_46872x62.csv?download=1
- Tox
    - references:
        - https://www.sciencedirect.com/science/article/pii/S0141813025079565
    - dataset https://raw.githubusercontent.com/cosylabiiit/MultiTox/refs/heads/main/Data/toxin3052.csv
