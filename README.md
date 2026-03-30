# Sample Selection Bias — Replication

This repository contains replication code for:

> Brewer, D., & Carlson, A. (2024). *Addressing Sample Selection Bias for 
> Machine Learning Methods*. Journal of Applied Econometrics, 39(3), 383–400.

## Authors

Vladislava Anashkina, Mona Bennis, and Robert Campbell Powers

## Overview

This repository replicates the section of Brewer & Carlson (2024) that 
revisits the simulation study of Huang et al. (2006). The original replication 
files provided by Brewer & Carlson did not include the `betaKMM` function 
required to reproduce the KMM weighting results. We reconstructed this function 
from scratch based on the quadratic program described in Huang et al. (2006), 
implementing it in MATLAB using an RBF kernel matrix and solved via `quadprog`.

## Files

- `SSML_huangrep.m` — main replication script for the Huang et al. simulation
- `betaKMM.m` — our implementation of the KMM weight estimation function

## Reference

Huang, J., Gretton, A., Borgwardt, K., Schölkopf, B., & Smola, A. (2006). 
*Correcting Sample Selection Bias by Unlabeled Data*. Advances in Neural 
Information Processing Systems 19.
