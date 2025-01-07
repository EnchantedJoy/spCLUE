# spCLUE

A Contrastive Learning Approach to Unified Spatial Transcriptomics Analysis Across Single-Slice and Multi-Slice Data

## Overview

![Overview of spCLUE](./spCLUE.png)

## Requirements

We recommend you to install the following packages to run **spCLUE**.

- python==3.9.0
- torch==1.13.1
- numpy==1.23.5
- scanpy==1.9.3
- anndata==0.8.0
- rpy2==3.4.1
- pandas==1.5.3
- scipy==1.10.0
- scikit-learn==1.2.2
- tqdm==4.64.1
- matplotlib==3.7.0
- seaborn==0.12.2
- jupyter==1.0.0
- R==4.2.0
- mclust==6.0.0

You can install **spCLUE** with anaconda by:

```shell
conda create -n spCLUE python=3.9.0
conda activate spCLUE
pip install -r requirements.txt
```

## Tutorial

you can follow the tutorial provided as jupyter files to run spCLUE.

**NOTE:** the path of your dataset should be different with the tutorial. Change the path correctly to run spCLUE.

## Datasets

Here we provide links of datasets used by spCLUE.

- **DLPFC**: [*http://spatial.libd.org/spatialLIBD/*](http://spatial.libd.org/spatialLIBD/).
- **BRCA**: [*https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data*](https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data).
- **BARISTA**: [*https://spacetx.github.io/data.html*](https://spacetx.github.io/data.html)
- **slideSeqV2_mob**: [*https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary*](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary).
- **stereoSeq_mob**: [*https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data*](https://github.com/JinmiaoChenLab/SEDR_analyses/tree/master/data).
- **stereoSeq_mosta**: [*https://db.cngb.org/stomics/mosta/*](https://db.cngb.org/stomics/mosta/).
