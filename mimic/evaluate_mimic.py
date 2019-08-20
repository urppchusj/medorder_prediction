#%%[markdown]
# # Evaluate a trained model on a test set

#%%[markdown]
# ## Imports

#%%

import os
import sys
sys.path.append('./mimic/')
import pathlib
import pickle
from datetime import datetime
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from gensim.sklearn_api import W2VTransformer
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize

from components_mimic import (TransformedGenerator, check_ipynb, data,
                              neural_network, pse_helper_functions,
                              visualization)

#%%[markdown]
# ## Global variables

#%%[markdown]
# ### Save path
#
# SAVE_DIR specifies where the data from the trained
# model will be loaded. Must be a subdirectory of "model".
# Will continue saving there.

#%%
SAVE_DIR = '20190819-1805training'
save_path = os.path.join('mimic', 'model', SAVE_DIR)

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
# Load the preprocessed data of the evaluation set

#%%[markdown]
# #### Load the data

#%%

hp = joblib.load(os.path.join(save_path, 'hp.joblib'))
if len(hp) == 3:
	BATCH_SIZE, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM = hp
elif len(hp) == 4:
	_, BATCH_SIZE, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM = hp

d = data(subset='test')

d.load_data(get_profiles=False)

#%%[markdown]
# #### Make the data lists

#%%
_, targets, seqs, active_meds, depas, _, _, _, _ = d.make_lists(get_valid=False)

#%%[markdown]
# ### Word2vec embeddings
#
# Load the previously fitted word2vec pipeline

#%%
w2v = joblib.load(os.path.join(save_path, 'w2v.joblib'))

#%%[markdown]
# ### Profile state encoder (PSE)
#
# Load the previously fitted profile state encoder

#%%
phf = pse_helper_functions()
pse_pp = phf.pse_pp
pse_a = phf.pse_a

pse = joblib.load(os.path.join(save_path, 'pse.joblib'))

#%%[markdown]
# ### Label encoder
#
# Load the previously fitted label encoder

#%%
le = joblib.load(os.path.join(save_path, 'le.joblib'))

#%%[markdown]
# ### Neural network
#
# Load the fitted neural network

#%%[markdown]
# Get the variables necessary to train the model from
# the fitted scikit-learn pipelines.

#%%
w2v_step = w2v.named_steps['w2v']
pse_shape = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
output_n_classes = len(le.classes_)

#%%[markdown]
# Filter out previously unseen labels and count how many
# are discarded

#%%
pre_discard_n_targets = len(targets)
targets = [target for target in targets if target in le.classes_]
seqs = [seq for seq, target in zip(seqs, targets) if target in le.classes_]
active_meds = [active_med for active_med, target in zip(active_meds, targets) if target in le.classes_]
depas = [depa for depa, target in zip(depas, targets) if target in le.classes_]
post_discard_n_targets = len(targets)

print('Predicting on {} samples, {:.2f} % of original samples, {} samples discarded because of unseen labels.'.format(pre_discard_n_targets, 100*post_discard_n_targets/pre_discard_n_targets, pre_discard_n_targets-post_discard_n_targets))

#%%[markdown]
# #### Sequence generators

#%%
eval_generator = TransformedGenerator(w2v_step, pse, le, targets, seqs, active_meds, depas, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

#%%[markdown]
# #### Instantiate the model

#%%
n = neural_network()
model = tf.keras.models.load_model(os.path.join(save_path, 'model.h5'), custom_objects={'sparse_top10_accuracy':n.sparse_top10_accuracy, 'sparse_top30_accuracy':n.sparse_top30_accuracy})

#%%[markdown]
# #### Evaluate the model
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

#%%
results = model.evaluate_generator(eval_generator, verbose=verbose)
predictions = model.predict_generator(eval_generator, verbose=verbose)

#%%[markdown]
# #### Compute and print evaluation metrics

#%%
print('Evaluation on test set results')
print('Predictions for {} classes'.format(len(le.classes_)))
print('{} classes reprensented in targets'.format(len(set(targets))))
for metric, result in zip(model.metrics_names, results):
	print('Metric: {}   Score: {:.5f}'.format(metric,result))

#%%
prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
pl_series = pd.Series(prediction_labels)
f = sns.countplot(pl_series, order=pl_series.value_counts().index[:50])
f.set(xticklabels='', xlabel='classes')
plt.savefig(os.path.join(save_path, 'predictions_distribution.png'))
nb_predicted_classes = len(pl_series.value_counts().index)
print('Number of classes predicted on evaluation set: {}'.format(nb_predicted_classes))

#%%
cr = classification_report(targets, prediction_labels, output_dict=True)
cr_df = pd.DataFrame.from_dict(cr, orient='index')
cr_df.to_csv(os.path.join(save_path,  'eval_classification_report.csv'))

#%%
binary_labels = label_binarize(targets, le.classes_)
filtered_binary_labels = np.delete(binary_labels,np.where(~binary_labels.any(axis=0))[0], axis=1)
filtered_predictions = np.delete(predictions,np.where(~binary_labels.any(axis=0))[0], axis=1)
rocauc_we = roc_auc_score(filtered_binary_labels, filtered_predictions, average='weighted')
print('Weighted average ROC AUC score for present labels: {:.3f}'.format(rocauc_we))
print(cr_df.describe())

#%%
department_data = pd.read_csv('mimic/data/depas.csv')
department_data.set_index('ORIG_SERVICE', inplace=True)
department_cat_dict = department_data['CAT_SERVICE'].to_dict()
departments = [list(active_med[-1])[0] for active_med in active_meds]
categorized_departments = [department_cat_dict[department] for department in departments]
unique_categorized_departments = list(set(categorized_departments))
for department in unique_categorized_departments:
	print('Results for category: {}'.format(department))
	indices = [i for i, value in enumerate(categorized_departments) if value == department]
	selected_seq = [seqs[i] for i in indices]
	selected_active_meds = [active_meds[i] for i in indices]
	selected_depa = [depas[i] for i in indices]
	selected_targets = [targets[i] for i in indices]
	eval_generator = TransformedGenerator(w2v_step, pse, le, selected_targets, selected_seq, selected_active_meds, selected_depa, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)
	results = model.evaluate_generator(eval_generator, verbose=1)
	predictions = model.predict_generator(eval_generator, verbose=1)
	print('Evaluation results')
	print('Predicting on {} samples, {:.2f} % of {} samples.'.format(len(selected_targets), (100*len(selected_targets))/len(targets), len(targets)))
	print('Predictions for {} classes'.format(len(le.classes_)))
	print('{} classes reprensented in targets'.format(len(set(selected_targets))))
	for metric, result in zip(model.metrics_names, results):
		print('Metric: {}   Score: {:.5f}'.format(metric,result))
	prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
	binary_labels = label_binarize(selected_targets, le.classes_)
	filtered_binary_labels = np.delete(binary_labels,np.where(~binary_labels.any(axis=0))[0], axis=1)
	filtered_predictions = np.delete(predictions,np.where(~binary_labels.any(axis=0))[0], axis=1)
	p_we, r_we, _, _ = precision_recall_fscore_support(selected_targets, prediction_labels, average='weighted')
	print('Weighted average precision score for present labels: {:.3f}'.format(p_we))
	print('Weighted average recall score for present labels: {:.3f}'.format(r_we))
