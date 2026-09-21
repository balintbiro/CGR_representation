# CGR_representation
Check whether the method for placing vertices during Chaos Game Representation (CGR) has an impact on biological sequence classification performance.

# Upon usage, please cite
```bib
@article{biro2026vertex,
  title={Vertex assignment for frequency chaos game representation of proteins affects classification performance},
  author={Bir{\'o}, B{\'a}lint and Subicz, T{\'\i}mea Kata and Barta, R{\'o}bert and Hartai, Soma S{\'a}ndor and Hiripi, L{\'a}szl{\'o} and Hoffmann, Orsolya Ivett},
  journal={Computational and Structural Biotechnology Journal},
  year={2026},
  publisher={AAAS}
}
```

## Datapreparation
The following script performs data gathering and cleaning of the specified datasets (DeepLoc, PFAM, Tox and Immune. Please scroll down for [references](#benchmark-datasets)). All the sequences that contain non proteinogenic amino acids were removed. The script creates a `tempfile` for a particular datasets which is being removed after data cleaning. The output file is now can be incorporated into training cycles.

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

The general schema is that random encodings (n=1000) were searched for all the datasets. ÍThis can be done using the [search script](#random-search) The following script performs the filtering and sorting of these results and performs FCGR data generation for the dedicated encodings (min,q1,q2,q3 and max). Since there is no correlation between encodings through different datasets, dedicated encodings are dataset specific. This is why the script creates a `dataset` and `model` specific encodings, `pfam` and `custom` in the example below.
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

To quantify training stochasticity, please run the following script with the proper dataset. It runs classification with a particular encoding but with varying seeds when initializing the CNN. It automatically runs a custom and ResNet18 CNNs. This data should be compared with the results of [random search](#random-search). For the publication, `--n-seed 1000` setting was used.
```shell
python -m codes.training_noise \
    --logfile results/logs/noise_quant.log \
    --seqfile data/immune_clean.csv \
    --outfile results/noise_quant.csv \
    --n-seed 2 \
    --task binary \
    --dataset_name immune
```

Based on the results of [random search](#random-search), the dedicated encodings should be generated. This is necessary for for the downstream analyses (cross validation and augmentation). Since there is no correlation between the random search results of the custom made and ResNet18 CNNs, the dedicated encoding generation should be performed for every dataset and CNN combination.
```shell
python -m codes.datapreparation.dedicated_encodings \
    --logfile results/logs/dedicated_encodings.log \
    --model custom \
    --dataset_name immune \
    --outdir results/dedicated_encodings/
```

## Random search
Running random search of different CGR/FCGR encodings. This script calls FCGR_gen.R from inside which is responsible for generating FCGR encodings. One call does searching with a custom convolutional nerual network (CNN) and ResNet18 too. The reported classification matrics are AUROC and F1 scores.
```shell
python -m codes.cnn_random_encoding_search.py \
    --logfile results/logs/cnn.log \
    --seqfile data/{dataset_name}_clean.csv \
    --outfile results/cnn_res_iter.csv \
    --task binary \
    --dataset_name {dataset_name} \
    --n 10_000
```
The resulting file is `results/cnn_res_iter.csv`.

## Dedicated encodings related experiments (CV and augmentation)
Once the dedicated encodings are generated, CV and augmentation can be performed. CV is dataset and encoding specific so it should be run separately for all the dataset and encoding combinations.
```shell
python -m codes.cnn_cv \
    --logfile results/logs/cv.log \
    --fcgr_matrix results/dedicated_encodings/pfam/custom/max.csv \
    --outfile results/cv_results.csv \
    --name pfam \
    --task binary \
    --model custom \
    --rank max \
    --res 35 \
    --n 10
```
Min encod-ing across benchmark datasets was augmented for training,either with a random sample from all 4 other encodings pooledtogether (`mix strategy`) or using one alternate encoding at a time (`single strategy`).
```shell
python -m codes.augmentation \
    --logfile results/logs/augmentation.log \
    --data_dir results/dedicated_encodings/deeploc/custom/ \
    --outfile results/augmentation_results.csv \
    --mix \
    --name deeploc \
    --model custom \
    --task binary \
    --n 2
```

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
