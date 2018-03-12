# Ensemble
C++ implementation of TransE, RESCAL-ALS, RESCAL_Rank, HOLE and Ensemble.

### Requirements

- Ubuntu 16.04
- GCC 5.4
- Boost 1.63
- Intel MKL 2017
- Intel TBB 4.4~20151115
- Blas 3.6.0

The component for logistic regression is modified from [Multi-core LIBLINEAR 2.11](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/) (released in March 2017).

### Data

- FB15k & WN18. Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, Oksana Yakhnenko, *Translating Embeddings for Modeling Multi-relational Data*. NIPS 2013. [[Download]](https://everest.hds.utc.fr/doku.php?id=en:transe)

### Evaluation

For fast evaluation, comment #define detailed_eval in include/tf/util/Base.h, so that only filtered hit@10, mean rank, MAP and MRR will be outputed. Otherwise, raw measures and evaluation for each relation will be outputed.

