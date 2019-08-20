#%%[markdown]
# # Start training the model WITHOUT a validation set (use full training set).

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from components_mimic import (TransformedGenerator, check_ipynb, data,
                        neural_network, pse_helper_functions, visualization)

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
SAVE_PATH = os.path.join(os.getcwd(), 'mimic', 'model', SAVE_STAMP + 'final_training')
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
N_TRAINING_EPOCHS = 13

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
# #### Save the hyperparameters

#%%
joblib.dump((N_TRAINING_EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM), os.path.join(SAVE_PATH, 'hp.joblib'))

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
# Prepare the data

#%%[markdown]
# #### Load the data
#
# If data restriction is used, saved the list of sampled encounters so that
# they can be reloaded to continue training with the same data if interrupted.

#%%
d = data()
d.load_data(restrict_data=RESTRICT_DATA, restrict_sample_size=RESTRICT_SAMPLE_SIZE)
if RESTRICT_DATA:
	with open(os.path.join(SAVE_PATH, 'sampled_encs.pkl'), mode='wb') as file:
		pickle.dump(d.enc, file)

#%%[markdown]
# #### Make the data lists

#%%
profiles_train, targets_train, seq_train, active_meds_train, depa_train, _, _, _, _ = d.make_lists(get_valid=False)

#%%[markdown]
# ### Word2vec embeddings
#
# Create a scikit-learn pipeline to train word2vec embeddings
# with training set profiles

#%%[markdown]
# #### Define, train and save the pipeline

#%%
w2v = Pipeline([
	('w2v', W2VTransformer(alpha=W2V_ALPHA, iter=W2V_ITER, size=W2V_EMBEDDING_DIM, hs=W2V_HS, sg=W2V_SG, min_count=W2V_MIN_COUNT, workers=W2V_WORKERS)),
	])

print('Fitting word2vec embeddings...')
w2v.fit(profiles_train)
w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)
joblib.dump(w2v, os.path.join(SAVE_PATH, 'w2v.joblib'))

#%%[markdown]
# ### Profile state encoder (PSE)
#
# Encode the profile state, composed of active meds, active pharmacological
# classes and department into a multi-hot vector

#%%[markdown]
# #### PSE helper functions

#%%
phf = pse_helper_functions()
pse_pp = phf.pse_pp
pse_a = phf.pse_a

#%%[markdown]
# #### Prepare the data for the pipeline, fit and save the PSE encoder to the training set

#%%
print('Preparing data for PSE...')
pse_data = [[ap, de] for ap, de in zip(active_meds_train, depa_train)]
n_pse_columns = len(pse_data[0])

pse_transformers = []
for i in range(n_pse_columns):
	pse_transformers.append(('pse{}'.format(i), CountVectorizer(binary=True, lowercase=False, preprocessor=pse_pp, analyzer=pse_a), i))
pse_pipeline_transformers = [
	('columntrans', ColumnTransformer(transformers=pse_transformers))
	]
pse = Pipeline(pse_pipeline_transformers)

print('Fitting PSE...')
pse.fit(pse_data)

joblib.dump(pse, os.path.join(SAVE_PATH, 'pse.joblib'))

#%%[markdown]
# ### Label encoder
#
# Encode the targets

#%%[markdown]
# #### Fit and save the label encoder to the train set targets

#%%
le = LabelEncoder()
le.fit(targets_train)

joblib.dump(le, os.path.join(SAVE_PATH, 'le.joblib'))

#%%[markdown]
# ### Neural network
#
# Train a neural network to predict each drug prescribed to a
# patient from the sequence of drug orders that came before
# it and the profile state excluding that drug.

#%%[markdown]
# Get the variables necessary to train the model from
# the fitted scikit-learn pipelines.

#%%
w2v_step = w2v.named_steps['w2v']
pse_shape = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
output_n_classes = len(le.classes_)

#%%[markdown]
# #### Sequence generators

#%%
train_generator = TransformedGenerator(w2v_step, pse, le, targets_train, seq_train, active_meds_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)

#%%[markdown]
# #### Instantiate the model

#%%
n = neural_network()
callbacks = n.callbacks(SAVE_PATH, callback_mode='train_no_valid')
model = n.define_model(LSTM_SIZE, N_LSTM, DENSE_PSE_SIZE, CONCAT_SIZE, DENSE_SIZE, DROPOUT, L2_REG, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, pse_shape, N_PSE_DENSE, N_DENSE, output_n_classes)
print(model.summary())
tf.keras.utils.plot_model(model, to_file=os.path.join(SAVE_PATH, 'model.png'))

#%%[markdown]
# #### Train the model
#
# Use in_ipynb to check if running in Jupyter or not to print
# progress bars if in terminal and log only at epoch level if
# in Jupyter. This is a bug of Jupyter or Keras where progress
# bars will flood stdout slowing down and eventually crashing 
# the notebook.

#%%
if in_ipynb:
	verbose=2
else:
	verbose=1

model.fit_generator(train_generator,
	epochs=N_TRAINING_EPOCHS,
	callbacks=callbacks,
	verbose=verbose)

model.save(os.path.join(SAVE_PATH, 'model.h5'))
