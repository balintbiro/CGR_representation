# CGR_representation
Check whether the method for placing vertices during Chaos Game Representation (CGR) has an impact on biological sequence classification perofrmance.

# Important articles:
- General CGR paper: https://www.sciencedirect.com/science/article/pii/S2001037021004736?via%3Dihub#b0010
- Grouping of nucleotides for Markov chains: https://www.sciencedirect.com/science/article/pii/S0025556406001003
- Grouping of amino acids: https://www.nature.com/articles/s41598-020-72174-5
- CGR for proteins: https://link.springer.com/article/10.1007/s00894-023-05777-0#article-info
- CGR and FCGR formulas: https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac119/6963321
- Right choice for random protein CGR resolution: "However, it can also lead to a compression if the length of the string is larger than the selected resolution" https://academic.oup.com/bioinformatics/article/36/1/272/5521624
- 0.865 as the optimal scaling factor for protein sequences: https://www.sciencedirect.com/science/article/abs/pii/0263785594801096?via%3Dihub (A. Fiser, G.E. Tusnady and I. Simon, Chaos game represenation of protein structures. J. Mol. Graphics, 12)
302-304, 1994
- Distance metrics for sequential data: https://pyts.readthedocs.io/en/latest/modules/metrics.html
  
_Why does it make sense to use different vertex assignments:_
    - Different vertex distribution should give the same result: "However, for long sequences, the CGR representation exhibits the property of self-similarity, i.e., a given pattern is repeated at different scales, for all three types of configurations" https://www.mdpi.com/1422-0067/23/3/1847
    - Data augmentation: Chaos Game Representations & Deep Learning for Proteome-Wide Protein Prediction (Downloaded to uni OneDrive)

```shell
# running FCGR generation
Rscript --vanilla FCGR_gen.R --encoding PSDNREQLCYFAIKVMGHWT --output_file data/PSDNREQLCYFAIKVMGHWT_hydrophobic_0865_35.csv --input_filename data/deeploc_clean.csv --scaling_factor 0.865 --resolution 35

# running random search of different CGR encodings. This script calls FCGR_gen.R from inside
python ./cnn_random_encoding_search.py --logfile cnn.log --seqfile data/deeploc_clean.csv --fcgrfile data/random_encoding_0865_35.csv --outfile data/cnn_res_iter.csv --sf 0.865 --res 35 --n 10_000

# running cross validation
python cnn_cv.py --logfile ../results/cv.log --fcgr_matrix ../data/q1.csv --outfile ../results/cnn_cv_q1.csv --res 35 --n 10

# running augmentation with cross validation
python augmentation.py --logfile augmentation.log --datasets_dir ../data/spec_datasets/ --outfile ../results/augmentation_results.csv --mix True
```

# Relationship between resolution and k-mers in FCGR. FCGR is produced by separate CGR by a grid. Let say the grid is 8*8 in this case 2^^k*2^^k.

# Benchmark datasets:
- Deeploc1.0 https://academic.oup.com/bioinformatics/article/33/21/3387/3931857
- GenomicBenchmarks OpenChromationRegion https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8
