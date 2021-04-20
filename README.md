# rng-scientific_validation

Scientific validation of new random number generators in NEST by means of statistical analysis of multi-area model spiking data.

- Figures in plots have been created by plot_statistics.py.
- See comments in that file for details on analysis performed.
- Key result files are
    - rngtest_{model}_{test}_{codeA}_{codeB}_{block1}_{block2}.[pdf|png]
    - rngtest_summary_{model}.[pdf|png]
    
- model: mam, 4x4
- test: KS (Kolmogorov-Smirnov), ES (Epps-Singleton)
- codeA, B: code version compared
- block1, 2: trial blocks compared

- summary: CDFs of p-values shown in rngtest_* figures.

Hans Ekkehard Plesser, 2021-04-20
