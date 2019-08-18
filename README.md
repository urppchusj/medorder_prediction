# Medorder prediction

Prediction of the next medication order to assist prescription verification by pharmacists in a health care center.

---

## Motivation

Health-system pharmacists review almost all medication orders for hospitalized patients. Considering that most orders contain no errors, especially in the era of CPOE with CDS,<sup>[1](https://doi.org/10.2146/ajhp060617)</sup> pharmacists have called for technology to enable the triage of routine orders to less extensive review, in order to focus pharmacist attention on unusual orders requiring.<sup>[2](https://doi.org/10.2146/ajhp070671),[3](https://doi.org/10.2146/ajhp080410),[4](https://doi.org/10.2146/ajhp090095)</sup>

We propose a machine learning model that learns medication order patterns from historical data, and then predicts the next medication order given a patient's previous order sequence, and currently active drugs. The predictions would be compared to actual orders. Our hypothesis is that orders ranking low in predictions would be unusual, while orders ranking high would be more likely to be routine or unremarkable.

## Description

This repository presents the code used to train and evaluate a machine learning model aiming to predict the next medication order during hospitalization. Because we are unable to share our dataset, we also provide a working implementation on [MIMIC-III](https://mimic.physionet.org/) as a demonstration (see caveats below). The model uses two inputs:

1. The sequence of previous drug orders, represented as word2vec embeddings.
2. The currently active drugs and pharmacological classes, as well as the ordering department, represented as a bag-of-words, transformed into a multi-hot vector through binary count vectorization.

The output is the prediction of the next medication order (the drug only, not the dose, route and frequency).

## Caveats about using the MIMIC dataset with our approach

**There are important limitations to using the MIMIC dataset with this approach as compared to to real, unprocessed data. We do not believe the results of the model on the MIMIC dataset would reliable enough for use in research or in practice. The mimic files should only be used for demonstration purposes.**

There are four main characteristics of the MIMIC dataset which make it less reliable for this model:

1. The MIMIC data only comes from ICU patients. As shown in our paper, ICU patients were one of the populations where our model showed the worst performance, probably because of the variability and the complexity of these patients. It would be interesting to explore how to increase the performance on this subset of patients, possibly by including more clinical data as features. Also, the same drugs may be used in patients in and out of ICU, and the information about the usage trends outside of ICU may be useful to the model, especially for the word2vec embeddings. This data is not present in MIMIC.
2. The MIMIC data was time-shifted inconsistently between patients. Medication use patterns follow trends over time, influcended by drug availability (new drugs on the market, drugs withdrawn from markets, drug shortages) and clinical pratices following new evidence being published. The inconsistent time-shifting destroys these trends; the dataset becomes effectiely pre-shuffled. In our original implementation, we were careful to use only the [Scikit-Learn TimeSeries Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) in cross-validation to preserve these patterns, and our test set was chronologically after our training-validation set and was preprocessed separately. In our MIMIC demonstration, we changed these to regular splits and our preprocessor generates the training-validation and test sets from a single file.
3. The MIMIC prescription data does not include the times of the orders, only the dates. This destroys the sequence of orders within a single day. In NLP, this would be equivalent to shuffling each sentence within a text and then trying to extract semantic information. This makes the creation of word2vec embeddings much more difficult because the exact context of the orders (i.e. those that occured immediately before and after) is destroyed. This shows in analogy accuracy which does not go above 20-25% on MIMIC while we achived close to 80% on our data. The sequence of orders, an important input to the model, is therefore inconsistent. Also, it becomes impossible to reliably know whether orders within the same day as the target were discontinued or not when the target was prescribed, descreasing the reliability of our multi-hot vector input.
4. The MIMIC dataset does not include the pharmacological classes of the drugs. The GSN (generic sequence number) allows linkage of the drug data to [First Databank](http://phsirx.com/blog/gpi-vs-gsn), which could allow extraction of classes, but this database is proprietary. One of our model inputs is therefore eliminated.

## Files

We present the files in the order they should be run to go from the original data files to a trained and evaluated model. The python script files are provided as commented files that [can be used to generate Jupyter Notebooks](https://code.visualstudio.com/docs/python/jupyter-support) or can be run as-is in the terminal. The MIMIC files are also provided directly as Jupyter notebooks.

The same files can be found in the `/paper` and the `/mimic` subdirectories. The `/paper` subdirectory shows the method we used on our dataset as described in our paper. These files could be adapted and used by other researchers to replicate our approach on data from other hospitals. The `/mimic` subdirectory is a working implementation on [MIMIC-III](http://dx.doi.org/10.1038/sdata.2016.35) as a demonstration taking into account the differences of the MIMIC data as compared to ours.

We present the files only once for both `/paper` and `/mimic`. Files in the `/mimic` subdirectory have and additional "mimic" in their names.

### preprocessor.py

This file is not formatted to generate a Jupyter notebook, can only be run as a script. 

The `/paper` file cannot be used without the original source data, but could be adapted to data from another source with the appropriate modifications. See the `/mimic` file for an example.

This file will transform the source files, which should be essentialy lists of orders and associated data, into several pickle files containing dictionaries where the keys are encounter ids and the values are the features for this encounter, in chronological order.

`enc_list.pkl` is a simple list of encounter ids, to allow for easy splitting into sets.
`profiles_list.pkl` is the list of the raw order sequences in each encounter, to train the word2vec embeddings.

After being loaded and processed by the data loader in `components.py`, each order gets considered as a label (`targets.pkl`). The features associated with this label are:
1. The sequence of orders preceding it within the encounter (`seq_list.pkl`). Orders happening at the exact same time are kept in the sequence. In MIMIC, because order times are precise to the day, this means each order that happened in the same day is present in the sequence (except the label).
2. The active drugs at the time the label was ordered (`active_meds_list.pkl`). Orders happening at the same time as the label are considered active.
3. The active pharmacological classes at the time the label was ordered (`active_classes_list.pkl`). This is not created by the MIMIC preprocessor.
4. The departement where the order happened (`depa_list.pkl`).

#### paper version

Arguments:
```
--sourcefile	indicates where the original data, in csv format, is located.
--definitionsfile	indiciates a separate file linking medication numbers to full medication names and pharmacological classes.
--numyears	indicates how many years of data to process from the file (starting from the most recent). Defaults to 5.
```

#### mimic version
Takes no arguments. Requires the ADMISSIONS.csv, PRESCRIPTIONS.csv and SERVICES.csv files from MIMIC-III in `/mimic/data/`.

### w2v_embeddings.py

Find the best word2vec training hyperparameters to maximize the accuracy on a list of analogies. We provide a list of analogies 


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
