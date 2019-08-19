#%%[markdown]
# # Cross-validate the model.
#
# Note that this code does not allow to stop and resume later.

#%%[markdown]
# ## Imports

#%%
import os
import pathlib
import pickle
from datetime import datetime
from multiprocessing import cpu_count

import joblib
import pandas as pd
import tensorflow as tf
from gensim.sklearn_api import W2VTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from components_mimic import (TransformedGenerator, check_ipynb, data,
                              neural_network, pse_helper_functions,
                              visualization)

#%%[markdown]
# ## Global variables

#%%[markdown]
# ### Save path
#
# Where everything will get saved. Will create a subdirectory
# called model with another subdirectory inside it with
# the date and time this block ran.

#%%
SAVE_STAMP = datetime.now().strftime('%Y%m%d-%H%M')
SAVE_PATH = os.path.join(os.getcwd(), 'mimic', 'model', SAVE_STAMP + 'crossval')
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

#%%[markdown]
# ### Data variables
#
# Set RESTRICT_DATA to True to execute with a sample of data instead
# of the entire dataset. Can be useful for faster execution when
# debugging or testing things.
# RESTRICT_SAMPLE_SIZE controls the number of encounters to sample
# when restricting the data.

#%%
RESTRICT_DATA = False
RESTRICT_SAMPLE_SIZE = 1000

#%%[markdown]
# ### Word2vec hyperparameters
#
# These are the best hyperparameters found for training word2vec
# embeddings as described in the paper. See Gensim documentation
# for explanation at https://radimrehurek.com/gensim/models/word2vec.html

#%%
W2V_ALPHA = 0.013
W2V_ITER = 32
W2V_EMBEDDING_DIM = 64
W2V_HS = 0
W2V_SG =  1
W2V_MIN_COUNT = 5
W2V_WORKERS = cpu_count()

#%%[markdown]
# ### Neural network hyperparameters
#
# These are the best hyperparameters found on our dataset as described
# in the paper.
#
# N_LSTM: Number of extra LSTM layers. Two layers are used at baseline. This
# number controls how many additional layers are used
# N_PSE_DENSE: Number of extra dense layers after the multi-hot vector input.
# One layer is used at baseline. This number controls how many
# additional layers are used
# N_DENSE: Number of dense layers used after concatenation. Baseline is one
# (the output layer). Each layer includes batch normalization, the 
# dense layer itself then dropout.

#%%
N_LSTM = 0 # total number of LSTM layers = this variable + 2
N_PSE_DENSE = 0 # total number of dense layers after multi-hot input = this number + 1
N_DENSE = 2 # total number of dense layers after concatenation = this number + 1 (the output layer)

LSTM_SIZE = 128
DENSE_PSE_SIZE = 128
CONCAT_SIZE = 256
DENSE_SIZE = 256
DROPOUT = 0.2
L2_REG = 0
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256

#%%[markdown]
# ## Execution

#%%[markdown]
# ### Jupyter notebook detection
#
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

#%%
in_ipynb = check_ipynb().is_inipynb()

#%%[markdown]
# ### Data
#
# Load the data

#%%
d = data()
d.load_data(restrict_data=RESTRICT_DATA, restrict_sample_size=RESTRICT_SAMPLE_SIZE)

#%%[markdown]
# ### PSE helper functions

#%%
phf = pse_helper_functions()
pse_pp = phf.pse_pp
pse_a = phf.pse_a

#%%[markdown]
# ### Cross-validation
#
# Do the cross validation. This is a single cell to avoid 
# problems if exeution stops. The code is commented to
# explain what is going on but refer to start_training.py
# for full details.

#%%
sizes_list = []
to_concat = []
split = 0
for train, val in ShuffleSplit(n_splits=5).split(d.enc):
	print('Performing cross-validation split: {}'.format(split))
	
	# prepare the data for the fold
	d.cross_val_split(train, val)
	profiles_train, targets_train, seq_train, active_meds_train, depa_train, targets_val, seq_val, active_meds_val, depa_val = d.make_lists()
	
	# train word2vec embeddings
	w2v = Pipeline([
	('w2v', W2VTransformer(alpha=W2V_ALPHA, iter=W2V_ITER, size=W2V_EMBEDDING_DIM, hs=W2V_HS, sg=W2V_SG, min_count=W2V_MIN_COUNT, workers=W2V_WORKERS)),
	])
	print('Fitting word2vec embeddings...')
	w2v.fit(profiles_train)
	w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)
	
	# fit the profile state encoder pipeline
	print('Fitting PSE...')
	pse_data = [[ap, de] for ap, de in zip(active_meds_train, depa_train)]
	n_pse_columns = len(pse_data[0])
	pse_transformers = []
	for i in range(n_pse_columns):
		pse_transformers.append(('pse{}'.format(i), CountVectorizer(binary=True, lowercase=False, preprocessor=pse_pp, analyzer=pse_a), i))
	pse_pipeline_transformers = [
		('columntrans', ColumnTransformer(transformers=pse_transformers))
		]
	pse = Pipeline(pse_pipeline_transformers)
	pse.fit(pse_data)
	
	# fit the label encoder
	le = LabelEncoder()
	le.fit(targets_train)
	
	# compute variables necessary to train the model from the
	# fitted scikit-learn pipelines
	w2v_step = w2v.named_steps['w2v']
	pse_shape = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
	output_n_classes = len(le.classes_)

	# sequence generators
	train_generator = TransformedGenerator(w2v_step, pse, le, targets_train, seq_train, active_meds_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)
	val_generator = TransformedGenerator(w2v_step, pse, le, targets_val, seq_val, active_meds_val, depa_val, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)
	n = neural_network()

	# instantiate the model
	callbacks = n.callbacks(SAVE_PATH, callback_mode='cross_val')
	model = n.define_model(LSTM_SIZE, N_LSTM, DENSE_PSE_SIZE, CONCAT_SIZE, DENSE_SIZE, DROPOUT, L2_REG, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, pse_shape, N_PSE_DENSE, N_DENSE, output_n_classes)

	# train the model
	if in_ipynb:
		verbose=2
	else:
		verbose=1
	
	print('Fitting neural network...')
	model.fit_generator(train_generator,
		epochs=1000,
		callbacks=callbacks,
		validation_data=val_generator,
		verbose=verbose)

	# Get the metrics for the best epoch (EarlyStopCallback restores best epoch)
	train_results = model.evaluate_generator(train_generator, verbose=verbose)
	val_results = model.evaluate_generator(val_generator, verbose=verbose)

	# Make a list of validation metrics names
	val_metrics = ['val_' + metric_name for metric_name in model.metrics_names]

	# Concatenate all results in a list
	all_results = [*train_results, *val_results]
	all_metric_names = [*model.metrics_names, *val_metrics]

	# make a dataframe with the fold metrics
	fold_results_df = pd.DataFrame.from_dict({split:dict(zip(all_metric_names, all_results))}, orient='index')
	sizes_list.append({'train':len(seq_train), 'val':len(seq_val)})
	to_concat.append(fold_results_df)
	split += 1

# Concatenate reults into a single dataframe
cv_results_df = pd.concat(to_concat)
cv_results_df.to_csv(os.path.join(SAVE_PATH, 'cv_results.csv'))

#%%[markdown]
# #### Plot the loss and accuracy of the cross-validation

#%%
v = visualization()

v.plot_crossval_accuracy_history(cv_results_df, SAVE_PATH)
v.plot_crossval_loss_history(cv_results_df, SAVE_PATH)
