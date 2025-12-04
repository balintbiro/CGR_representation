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

```python
side_groups={
    'A':[-1,1],
    'C':[-1,-1],
    'G':[1,-1],
    'T':[1,1]
}
structure={
    'A':[-1,1],
    'G':[-1,-1],
    'C':[1,-1],
    'T':[1,1]
}
bonds={
    'A':[-1,1],
    'T':[-1,-1],
    'G':[1,-1],
    'C':[1,1]
}
```
Relationship between resolution and k-mers in FCGR. FCGR is produced by separate CGR by a grid. Let say the grid is 8*8 in this case 2^^k*2^^k.

# Benchmark datasets:
- Deeploc1.0 https://academic.oup.com/bioinformatics/article/33/21/3387/3931857
- GenomicBenchmarks OpenChromationRegion https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8
