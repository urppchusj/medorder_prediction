Prediction of the next medication order to assist prescription verification by pharmacists in a health care center.

---

# Motivation

Health-system pharmacists review almost all medication orders for hospitalized patients. Considering that most orders contain no errors, especially in the era of CPOE with CDS,<sup>[1](https://doi.org/10.2146/ajhp060617)</sup> pharmacists have called for technology to enable the triage of routine orders to less extensive review, in order to focus pharmacist attention on unusual orders requiring.<sup>[2](https://doi.org/10.2146/ajhp070671),[3](https://doi.org/10.2146/ajhp080410),[4](https://doi.org/10.2146/ajhp090095)</sup>

We propose a machine learning model that learns medication order patterns from historical data, and then predicts the next medication order given a patient's previous order sequence, and currently active drugs. The predictions would be compared to actual orders. Our hypothesis is that orders ranking low in predictions would be unusual, while orders ranking high would be more likely to be routine or unremarkable.

# Description

This repository presents the code used to train and evaluate a machine learning model aiming to predict the next medication order during hospitalization. Because we are unable to share our dataset, we also provide a working implementation on [MIMIC-III](https://mimic.physionet.org/) as a demonstration (see caveats below). The model uses two inputs:

1. The sequence of previous drug orders, represented as word2vec embeddings.
2. The currently active drugs and pharmacological classes, as well as the ordering department, represented as a bag-of-words, transformed into a multi-hot vector through binary count vectorization.

The output is the prediction of the next medication order (the drug only, not the dose, route and frequency).

## Caveats about using the MIMIC dataset with our approach

**There are important limitations to using the MIMIC dataset with this approach as compared to to real, unprocessed data. We do not believe the results of the model on the MIMIC dataset would reliable enough for use in research or in practice. The mimic files should only be used for demonstration purposes.**

There are four main characteristics of the MIMIC dataset which make it less reliable for this model:

1. The MIMIC data only comes from ICU patients. As shown in our paper, ICU patients were one of the populations where our model showed the worst performance, probably because of the variability and the complexity of these patients. It would be interesting to explore how to increase the performance on this subset of patients, possibly by including more clinical data as features. Also, the same drugs may be used in patients in and out of ICU, and the information about the usage trends outside of ICU may be useful to the model, especially for the word2vec embeddings. This data is not present in MIMIC.
2. The MIMIC data was time-shifted inconsistently between patients. Medication use patterns follow trends over time, influcended by drug availability (new drugs on the market, drugs withdrawn from markets, drug shortages) and clinical pratices following new evidence being published. The inconsistent time-shifting destroys these trends; the dataset becomes effectiely pre-shuffled. In our original implementation, we were careful to use only the [Scikit-Learn TimeSeries Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) in cross-validation to preserve these patterns, and our test set was chronologically after our training-validation set and was preprocessed separately. In our MIMIC demonstration, we changed these to regular shuffle splits and our preprocessor generates the training-validation and test sets from the raw MIMIC files.
3. The MIMIC prescription data does not include the times of the orders, only the dates. This destroys the sequence of orders within a single day. In NLP, this would be equivalent to shuffling each sentence within a text and then trying to extract semantic information. This makes the creation of word2vec embeddings much more difficult because the exact context of the orders (i.e. those that occured immediately before and after) is destroyed. This shows in analogy accuracy which does not go above 20-25% on MIMIC while we achived close to 80% on our data. The sequence of orders, an important input to the model, is therefore inconsistent. Also, it becomes impossible to reliably know whether orders within the same day as the target were discontinued or not when the target was prescribed, descreasing the reliability of our multi-hot vector input.
4. The MIMIC dataset does not include the pharmacological classes of the drugs. The GSN (generic sequence number) allows linkage of the drug data to [First Databank](http://phsirx.com/blog/gpi-vs-gsn), which could allow extraction of classes, but this database is proprietary. One of our model inputs is therefore eliminated.

# Files

We present the files in the order they should be run to go from the original data files to a trained and evaluated model. The python script files are provided as commented files that [can be used to generate Jupyter Notebooks](https://code.visualstudio.com/docs/python/jupyter-support) or can be run as-is in the terminal. Some files are also provided directly as Jupyter Notebooks.

The same files can be found in the `/paper` and the `/mimic` subdirectories. The `/paper` subdirectory shows the method we used on our dataset as described in our paper. These files could be adapted and used by other researchers to replicate our approach on data from other hospitals. The `/mimic` subdirectory is a working implementation on [MIMIC-III](http://dx.doi.org/10.1038/sdata.2016.35) as a demonstration taking into account the differences of the MIMIC data as compared to ours.

We present the files only once for both `/paper` and `/mimic`. Files in the `/mimic` subdirectory have and additional "mimic" in their names.

## preprocessor.py

This file is not formatted to generate a Jupyter notebook, can only be run as a script. 

The `/paper` file cannot be used without the original source data, but could be adapted to data from another source with the appropriate modifications. See the `/mimic` file for an example.

This script will transform the source files, which should be essentialy lists of orders and associated data, into several pickle files containing dictionaries where the keys are encounter ids and the values are the features for this encounter, in chronological order.

`enc_list.pkl` is a simple list of encounter ids, to allow for easy splitting into sets.
`profiles_list.pkl` is the list of the raw order sequences in each encounter, to train the word2vec embeddings.

After being loaded and processed by the data loader in `components.py`, each order gets considered as a label (`targets.pkl`). The features associated with this label are:
1. The sequence of orders preceding it within the encounter (`seq_list.pkl`). Orders happening at the exact same time are kept in the sequence. In MIMIC, because order times are precise to the day, this means each order that happened in the same day is present in the sequence (except the label).
2. The active drugs at the time the label was ordered (`active_meds_list.pkl`). Orders happening at the same time as the label are considered active.
3. The active pharmacological classes at the time the label was ordered (`active_classes_list.pkl`). This is not created by the MIMIC preprocessor.
4. The departement where the order happened (`depa_list.pkl`).

### paper version

Arguments:
```
--sourcefile	indicates where the original data, in csv format, is located.
--definitionsfile	indiciates a separate file linking medication numbers to full medication names and pharmacological classes.
--numyears	indicates how many years of data to process from the file (starting from the most recent). Defaults to 5.
```

### mimic version
Takes no arguments. Requires the ADMISSIONS.csv, PRESCRIPTIONS.csv and SERVICES.csv files from MIMIC-III in `/mimic/data/`.

## w2v_embeddings.py

Find the best word2vec training hyperparameters to maximize the accuracy on a list of analogies. We provide a list of pairs for the mimic dataset where the semantic relationship is going from a drug in tablet form to a drug in oral solution form (`mimic/data/pairs.txt`), as described in our paper. The file `utils/w2v_analogies.py` transforms these pairs into an analogy file (`mimic/data/eval_analogy.txt`) matching specifications for the [gensim accuracy evaluation method](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors.accuracy) that is used for scoring.

The script performs grid search with 3-fold cross-validation to explore the hyperparameter space, and then refits on the whole data with the best hyperparameters returns the analogy accuracy on the entire dataset. Clustering on 3d UMAP projected word2vec embeddings is explored to qualitatively evaluate if clusters correlate to clinical concepts and a 3d plot is returned showing the 3d projected embeddings with color-coded clusters. The clustering part is not used in the subsequent neural network.

We provide a Jupyter Notebook showing our summary exploration of the hyperparameter space on the MIMIC dataset. Performance is poor on this dataset, see caveats above.

Once the word2vec hyperparameters are found, they should be adjusted in the `cross_validate.py`, `start_training_with_valid.py` and `start_training_no_valid.py` files.

## cross_validate.py

This script performs 5-fold cross-validation of the neural network predicting the next medication order. The word2vec embeddings, multi-hot vector and label encoders are retrained at each fold using only the training data for that fold, to avoid data leakage between folds. Training uses a decreasing learning rate followed by early stopping to stop training at each fold when the validation loss stops decrasing. The best epoch is then restored and fold metrics are computed on that best epoch. A plot showing the training and validation accuracies and losses at each fold is produced.

This script should be used to explore the hyperparameter space and configuration of the neural network.

This script contains a `RESTRICT_DATA` flag that can be enabled to use only a sample of orders instead of the whole dataset. The sample size can be adjusted. This can be useful to try different things and to debug faster.

## start_training_with_valid.py

Once the best network configuration and hyperparameters have been found with cross-validation, this script performs training without cross-validation, but with a single validation set. This script should be used to explore the training process, and especially to determine for how many epochs to train and how to schedule the learning rate decrase for the final model. This script saves a checkpoint after each epoch so that training can be resumed if interrupted. Then, the best epoch is restored and this model is saved.

The number of epochs to train the final model should be adjusted in `start_training_no_valid.py`. The learning rate decrease schedule should be adjusted in components.py, under the `schedule` function in the `neural_network` class.

This script contains a `RESTRICT_DATA` flag that can be enabled to use only a sample of orders instead of the whole dataset. The sample size can be adjusted. This can be useful to try different things and to debug faster.

## resume_training_with_valid.py

This script can be used to resume training with validation if interrupted.

## start_training_no_valid.py

Once the number of training epochs and the learning rate schedule have been determined, this script is used to train the final model.  This script saves a checkpoint after each epoch so that training can be resumed if interrupted.

This script contains a `RESTRICT_DATA` flag that can be enabled to use only a sample of orders instead of the whole dataset. The sample size can be adjusted. This can be useful to try different things and to debug faster.

## resume_training_no_valid.py

This script can be used to resume training with no validation if interrupted.

## evaluate.py

This script uses the test subset of the data to determine the final performance metrics of the model. Will compute global metrics and metrics by patient category, as described in the paper. To categories are determined by grouping similar departments together. 

In the MIMIC dataset, this is based on the `SERVICES` table. Because the services transfers are precise to the second, but medication data is precise to the day, we approximated that the last service of the day was the prescriber of all orders within that day. Therefore, there is some degree of noise in the computed metrics. We provide department groupings following categories in line with what we used in our paper. To adjust how departments are grouped together, see the `mimic/data/depas.csv` file.

## components.py

This file is not meant to be run, but is a collection of classes and functions used in the previous scripts. The neural network architecture can be adjusted within this file.

## utils/dummy_class.py

This script acts as a dummy classifier, calculating accuracy metrics on the MIMIC training set if the top1, top10 and top30 most popular drugs in the dataset were always predicted. It also returns the number of classes within that set and a frequency histogram of the 50 most popular classes.

## utils/extract_druginfo_mimic.py

This script extracts the drug information from the MIMIC `PRESCRIPTIONS` table to a `definitions.csv` file providing the formulary drug code in relation to a computed string including the drug name, product strength and pharmaceutical form. This can be useful to match formulary drug codes to human-readable strings. Be careful, some formulary codes match to multiple strings.

## utils/w2v_analogies.py

This script takes a list of pairs of drugs and computes analogies, see `w2v_embeddings` above.

# Prerequisites

Developed using Python 3.7

Requires:

- Joblib
- Numpy
- Pandas
- Scikit-learn
- Scikit-plot
- UMAP
- Matplotlib
- Seaborn
- Gensim
- Tensorflow 1.13 or later
- Jupyter

# Contributors

Maxime Thibault.

# References

Paper currently under peer review for publication.  
Abstract presented at the Machine Learning for Healthcare 2019 conference  
[Abstract](https://www.mlforhc.org/accepted-papers) (spotlight session 6 abstract 2)  
[Spotlight presentation](https://www.ustream.tv/recorded/123483908) (from 52:15 to 54:15)  
[Poster](http://bit.ly/2JNGNP2)

# License

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
