# CGR_representation
Check whether the method for generating Chaos Game Representation (CGR) has an impact on nucleic acid classification perofrmance.

# Two very important articles:
- General CGR paper: https://www.sciencedirect.com/science/article/pii/S2001037021004736?via%3Dihub#b0010
- Grouping of nucleotides for Markov chains: https://www.sciencedirect.com/science/article/pii/S0025556406001003

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
