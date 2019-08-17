# Medorder prediction

Prediction of the next medication order to assist prescription verification by pharmacists in a health care center.

---

## Motivation

Health-system pharmacists review almost all medication orders for hospitalized patients. Considering that most orders contain no errors, especially in the era of CPOE with CDS,<sup>[1](https://doi.org/10.2146/ajhp060617)</sup> pharmacists have called for technology to enable triaging routine orders to less extensive review, in order to focus pharmacist attention on unusual orders requiring.[2](https://doi.org/10.2146/ajhp070671),[3](https://doi.org/10.2146/ajhp080410),[4](https://doi.org/10.2146/ajhp090095)

We propose a machine learning model that learns medication order patterns from historical data, and then predicts the next medication order given a patient's previous order sequence, and currently active drugs. The predictions would be compared to actual orders. Our hypothesis is that orders ranking low in predictions would be unusual, while orders ranking high would be more likely to be routine or unremarkable.

## Description

This repository presents the code used to train and evaluate a machine learning model aiming to predict the next medication order during hospitalization. Because we are unable to share our dataset, we also provide a working implementation on [MIMIC-III](https://mimic.physionet.org/) as a demonstration (see caveats below). The model uses two inputs:

1. The sequence of previous drug orders, represented as word2vec embeddings.
2. The currently active drugs and pharmacological classes, as well as the ordering department, represented as a bag-of-words, transformed into a multi-hot vector through binary count vectorization.

The output is the prediction of the next medication order (the drug only, not the dose, route and frequency).

## Files

We present the files in the order they should be run to go from the original data files to a trained and evaluated model. The python script files are provided as commented files that [can be used to generate Jupyter Notebooks](https://code.visualstudio.com/docs/python/jupyter-support) or can be run as-is in the terminal. The MIMIC files are also provided directly as Jupyter notebooks.

The same files can be found in the `/paper` and the `/mimic` subdirectories. The `/paper` subdirectory shows the method we used on our dataset as described in our paper. These files could be adapted and used by other researchers to replicate our approach on data from other hospitals. The `/mimic` subdirectory is a working implementation on [MIMIC-III](http://dx.doi.org/10.1038/sdata.2016.35) as a demonstration taking into account the differences of the MIMIC data as compared to ours. **There are important limitations to using the MIMIC dataset with this approach as compared to to real, unprocessed data. We do not believe the results of the model on the MIMIC dataset would reliable enough for use in research or in practice (see below). These files should only be used for demonstration purposes.**

### preprocess

To be completed

### Caveats about using the MIMIC dataset with our approach

To be completed

## Prerequisites

Developed using Python 3.7

Requires:

- To be completed

## Contributors

Maxime Thibault.

## License

GNU GPL v3

Copyright (C) 2019 Maxime Thibault

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
