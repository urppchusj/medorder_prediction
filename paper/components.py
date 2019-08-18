import os
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras, test


class check_ipynb:

	'''
	Verifies if current execution is in a Jupyter Notebook.
	Prints result and returns True if in Jupyter Notebook, else false.
	'''

	def __init__(self):
		pass

	def is_inipynb(self):
		try:
			get_ipython()
			print('Execution in Jupyter Notebook detected.')
			return True
		except:
			print('Execution outside of Jupyter Notebook detected.')
			return False


class data:

	'''
	Functions related to data loading and preparation.
	'''

	def __init__(self, datadir):
		# Prepare the data paths given a directory.
		# File names correspond to what is created by the preprocessor.
		self.profiles_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'profiles_list.pkl')
		self.targets_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'targets_list.pkl')
		self.seq_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'seq_list.pkl')
		self.activemeds_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'active_meds_list.pkl')
		self.activeclasses_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'active_classes_list.pkl')
		self.depa_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'depa_list.pkl')
		self.enc_file = os.path.join(os.getcwd(), 'paper', 'preprocessed_data', datadir, 'enc_list.pkl')

	def load_data(self, restrict_data=False, restrict_sample_size=None, previous_encs_path=False, get_profiles=True):
		
		# This allows to prevent loading profiles for nothing when
		# resuming training and word2vec embeddings do not need
		# to be retrained.
		if get_profiles:
			print('Loading profiles...')
			with open(self.profiles_file, mode='rb') as file:
				self.profiles = pickle.load(file)
		else:
			self.profiles=None
		# Load all the files
		print('Loading targets...')
		with open(self.targets_file, mode='rb') as file:
			self.targets = pickle.load(file)
		print('Loading sequences...')
		with open(self.seq_file, mode='rb') as file:
			self.seqs = pickle.load(file)
		print('Loading active meds...')
		with open(self.activemeds_file, mode='rb') as file:
			self.active_meds = pickle.load(file)
		print('Loading active classes...')
		with open(self.activeclasses_file, mode='rb') as file:
			self.active_classes = pickle.load(file) 
		print('Loading departments...')
		with open(self.depa_file, mode='rb') as file:
			self.depas = pickle.load(file) 
		print('Loading encounters...')
		# If reloading in order to resume training, this allows to reload
		# the same set of encounters when encounters have been sampled
		# using the restrict_data flag, to resume training with exactly
		# the same.
		if previous_encs_path:
			self.enc_file = previous_encs_path
		with open(self.enc_file, mode='rb') as file:
				self.enc = pickle.load(file)
		# The restrict_data flags allows to sample a number of encounters
		# defined by restrict_sample_size, to allow for faster execution
		# when testing code.
		if restrict_data:
			print('Data restriction flag enabled, sampling {} encounters...'.format(restrict_sample_size))
			self.enc = [self.enc[i] for i in sorted(random.sample(range(len(self.enc)), restrict_sample_size))]
		
	def split(self):
		print('Splitting encounters into train and test sets...')
		self.enc_train, self.enc_test = train_test_split(self.enc, shuffle=False, test_size=0.25)

	def cross_val_split(self, train_indices, test_indices):
		print('Splitting encounters into train and test sets...')
		self.enc_train = [self.enc[i] for i in train_indices]
		self.enc_test = [self.enc[i] for i in test_indices]

	def make_lists(self, get_test=True):
		print('Building data lists...')

		# Training set
		print('Building training set...')
		# If the get_test flag is set to False, put all encounters
		# in the training set. This can be used for evaluation
		# of a trained model, in this case the "training" set is
		# actually an evaluation set that does not get split.
		if get_test == False:
			self.enc_train = self.enc
		# Allocate profiles only if they have been loaded
		if self.profiles != None:
			self.profiles_train = [self.profiles[enc] for enc in self.enc_train]
		else:
			self.profiles_train=[]
		self.targets_train = [target for enc in self.enc_train for target in self.targets[enc]]
		self.seq_train = [seq for enc in self.enc_train for seq in self.seqs[enc]]
		self.active_meds_train = [active_med for enc in self.enc_train for active_med in self.active_meds[enc]]
		self.active_classes_train = [active_class for enc in self.enc_train for active_class in self.active_classes[enc]]
		self.depa_train = [str(depa) for enc in self.enc_train for depa in self.depas[enc]]

		# Make a list of unique targets in train set to exclude unseen targets from test set
		unique_targets_train = list(set(self.targets_train))

		# Test set is built only if necessary
		if get_test:
			print('Building test set...')
			# Filter out samples with previously unseen labels.
			self.targets_test = [target for enc in self.enc_test for target in self.targets[enc] if target in unique_targets_train]
			self.seq_test = [seq for enc in self.enc_test for seq, target in zip(self.seqs[enc], self.targets[enc]) if target in unique_targets_train]
			self.active_meds_test = [active_med for enc in self.enc_test for active_med, target in zip(self.active_meds[enc], self.targets[enc]) if target in unique_targets_train]
			self.active_classes_test = [active_class for enc in self.enc_test for active_class, target in zip(self.active_classes[enc], self.targets[enc]) if target in unique_targets_train]
			self.depa_test = [str(depa) for enc in self.enc_test for depa, target in zip(self.depas[enc], self.targets[enc]) if target in unique_targets_train]
		else:
			self.targets_test = None
			self.seq_test = None
			self.active_meds_test = None
			self.active_classes_test = None
			self.depa_test = None
		
		# Shuffle the training set.
		print('Shuffling training set...')
		shuffled = list(zip(self.targets_train, self.seq_train, self.active_meds_train, self.active_classes_train, self.depa_train))
		random.shuffle(shuffled)
		self.targets_train, self.seq_train, self.active_meds_train, self.active_classes_train, self.depa_train = zip(*shuffled)

		# Print out the number of samples obtained to make sure they match.
		print('Training set: Obtained {} profiles, {} targets, {} sequences, {} active meds, {} active classes, {} depas and {} encounters.'.format(len(self.profiles_train), len(self.targets_train), len(self.seq_train), len(self.active_meds_train), len(self.active_classes_train), len(self.depa_train), len(self.enc_train)))

		if get_test == True:
			print('Validation set: Obtained {} targets, {} sequences, {} active meds, {} active classes, {} depas and {} encounters.'.format(len(self.targets_test), len(self.seq_test), len(self.active_meds_test), len(self.active_classes_test), len(self.depa_test), len(self.enc_test)))

		return self.profiles_train, self.targets_train, self.seq_train, self.active_meds_train, self.active_classes_train, self.depa_train, self.targets_test, self.seq_test, self.active_meds_test, self.active_classes_test, self.depa_test


class pse_helper_functions:

	def __init__(self):
		pass

	# preprocessor (join the strings with spaces to simulate a text)
	def pse_pp(self, x):
		return ' '.join(x)

	# analyzer (do not transform the strings, use them as is because they are not words.)
	def pse_a(self, x):
		return x


class TransformedGenerator(keras.utils.Sequence):

	'''
	This is a Sequence generator that takes the fitted scikit-learn pipelines, the data
	and some parameters required to do the transformation properly, and them transforms
	them batch by batch into numpy arrays. This is necessary because transforming the 
	entire dataset at once uses an ungodly amount of RAM and takes forever.
	'''

	def __init__(self, w2v, pse, le, y, X_w2v, X_am, X_ac, X_depa, w2v_embedding_dim, sequence_length, batch_size, shuffle=True, return_targets=True):
		# Fitted scikit-learn pipelines
		self.w2v = w2v
		self.pse = pse
		self.le = le
		# Data
		self.y = y
		self.X_w2v = X_w2v
		self.X_am = X_am
		self.X_ac = X_ac
		self.X_depa = X_depa
		# Transformation parameters
		self.w2v_embedding_dim = w2v_embedding_dim
		self.sequence_length = sequence_length
		# Training hyperparameters
		self.batch_size = batch_size
		# Do you want to shuffle ? True if you train, False if you evaluate
		self.shuffle = shuffle
		# Do you want the targets ? True if you're training or evaluating,
		# False if you're predicting
		self.return_targets = return_targets

	def __len__(self):
		# Required by tensorflow, compute the length of the generator
		# which is the number of batches given the batch size
		return int(np.ceil(len(self.X_w2v) / float(self.batch_size)))

	def __getitem__(self, idx):
		# Transformation happens here.
		# Features go into a dict
		X = dict()
		# Transform the sequence into word2vec embeddings
		# Get a batch
		batch_w2v = self.X_w2v[idx * self.batch_size:(idx+1) * self.batch_size]
		# Transform if medication is in word2vec vocab, otherwize a zeros array shaped like word2vec embeddings
		transformed_w2v = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v]
		# Pad the sequences with zeros up to the sequence length.
		# Here it is SUPER important to indicate dtype='float32' otherwise the pad_sequences
		# function will transform everything into integers
		transformed_w2v = keras.preprocessing.sequence.pad_sequences(transformed_w2v, maxlen=self.sequence_length, dtype='float32')
		X['w2v_input']=transformed_w2v
		# Transform the active meds, pharmacological classes and department into a multi-hot vector
		# Get batches
		batch_am = self.X_am[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_ac = self.X_ac[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_depa = self.X_depa[idx * self.batch_size:(idx+1) * self.batch_size]
		# Prepare the batches for input into the ColumnTransformer step of the pipeline
		batch_pse = [[bm, bc, bd] for bm, bc, bd in zip(batch_am, batch_ac, batch_depa)]
		# Transform
		transformed_pse = self.pse.transform(batch_pse)
		# Output of the pipeline is a sparse matrix, convert to dense
		X['pse_input']=transformed_pse.todense()
		# Target
		if self.return_targets:
			# Get a batch
			batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
			# Transform the batch
			transformed_y = self.le.transform(batch_y)
			y = {'main_output': transformed_y}
			return X, y
		else:
			return X
	
	def on_epoch_end(self):
		# Shuffle after each training epoch so that the data is not always
		# seen in the same order
		if self.shuffle == True:
			shuffled = list(zip(self.y, self.X_w2v, self.X_am, self.X_ac, self.X_depa))
			random.shuffle(shuffled)
			self.y, self.X_w2v, self.X_am, self.X_ac, self.X_depa = zip(*shuffled)


class neural_network:

	'''
	Functions related to the neural network
	'''

	def __init__(self):
		pass

	# Custom accuracy metrics
	def sparse_top10_accuracy(self, y_true, y_pred):
		sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=10))

	def sparse_top30_accuracy(self, y_true, y_pred):
		sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=30))

	# Callbacks during training
	def callbacks(self, save_path, callback_mode='train_with_valid', n_done_epochs=0):

		# Assign simple names
		CSVLogger = keras.callbacks.CSVLogger
		EarlyStopping = keras.callbacks.EarlyStopping
		ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
		ModelCheckpoint = keras.callbacks.ModelCheckpoint
		LearningRateScheduler = keras.callbacks.LearningRateScheduler

		# Define the callbacks
		callbacks = []

		# Train with valid and cross-val callbacks
		if callback_mode == 'train_with_valid' or callback_mode == 'cross_val':
			callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.0005))
			callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, restore_best_weights=True))
		# Train with valid and train no valid callbacks
		if callback_mode == 'train_with_valid' or callback_mode == 'train_no_valid':
			callbacks.append(ModelCheckpoint(os.path.join(save_path, 'partially_trained_model.h5'), verbose=1))
			callbacks.append(CSVLogger(os.path.join(save_path, 'training_history.csv'), append=True))
		if callback_mode == 'train_no_valid':
			callbacks.append(LearningRateScheduler(self.schedule, verbose=1))
			callbacks.append(EpochLoggerCallback(save_path, n_done_epochs))
		return callbacks

	def schedule(self, i, cur_lr):
		# The schedule is hardcoded here from the results
		# of a training with validation
		new_lr = cur_lr
		return new_lr

	def define_model(self, sequence_size, n_add_seq_layers, dense_pse_size, concat_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, pse_shape, n_add_pse_dense, n_dense, output_n_classes):
		
		# Assign simple names
		# Use CuDNN implementation of LSTM if GPU is available, LSTM if it isn't
		# (non-CuDNN implementation is slower even on GPU)
		if test.is_gpu_available():
			LSTM = keras.layers.CuDNNLSTM
		else:
			LSTM = keras.layers.LSTM

		Dense = keras.layers.Dense
		Dropout = keras.layers.Dropout
		Input = keras.layers.Input
		BatchNormalization = keras.layers.BatchNormalization
		concatenate = keras.layers.concatenate
		l2 = keras.regularizers.l2
		Model = keras.models.Model

		# Define the neural network structure

		to_concat = []
		inputs= []
		
		# The word2vec inputs and layers before concatenation
		w2v_input = Input(shape=(sequence_length, w2v_embedding_dim, ), dtype='float32', name='w2v_input')
		w2v = LSTM(sequence_size, return_sequences=True)(w2v_input)
		w2v = Dropout(dropout)(w2v)
		for _ in range(n_add_seq_layers):
			w2v = LSTM(sequence_size, return_sequences=True)(w2v)
			w2v = Dropout(dropout)(w2v)
		w2v = LSTM(sequence_size)(w2v)
		w2v = Dropout(dropout)(w2v)
		w2v = Dense(sequence_size, activation='relu')(w2v)
		w2v = Dropout(dropout)(w2v)
		to_concat.append(w2v)
		inputs.append(w2v_input)
		
		# The multi-hot vector input (profile state encoder) and layers before concatenation
		pse_input = Input(shape=(pse_shape,), dtype='float32', name='pse_input')
		pse = Dense(dense_pse_size, activation='relu', kernel_regularizer=l2(l2_reg))(pse_input)
		pse = Dropout(dropout)(pse)
		for _ in range(n_add_pse_dense):
			pse = BatchNormalization()(pse)
			pse = Dense(dense_pse_size, activation='relu', kernel_regularizer=l2(l2_reg))(pse)
			pse = Dropout(dropout)(pse)
		to_concat.append(pse)
		inputs.append(pse_input)

		# Concatenation and dense layers
		concatenated = concatenate(to_concat)
		for _ in range(n_dense):
			concatenated = BatchNormalization()(concatenated)
			concatenated = Dense(concat_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated)
			concatenated = Dropout(dropout)(concatenated)
		concatenated = BatchNormalization()(concatenated)
		output = Dense(output_n_classes, activation='softmax', name='main_output')(concatenated)

		# Compile the model
		model = Model(inputs = inputs, outputs = output)
		model.compile(optimizer='Adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy', self.sparse_top10_accuracy, self.sparse_top30_accuracy])
		
		return model


class EpochLoggerCallback(keras.callbacks.Callback):

	'''
	Custom callback that logs done epochs so that training
	can be resumed and continue for the correct number of
	total epochs
	'''

	def __init__(self, save_path, n_done_epochs=0):
		self.save_path = save_path
		self.starting_epoch=n_done_epochs

	def on_epoch_end(self, epoch, logs=None):
		self.done_epochs=self.starting_epoch + epoch
		with open(os.path.join(self.save_path, 'done_epochs.pkl'), mode='wb') as file:
			pickle.dump(self.done_epochs, file)


class visualization:

	'''
	Functions that plot graphs
	'''

	def __init__(self):
		# Will be useful to decide to either show the plot (in
		# Jupyer Notebook) or save as a file (outside notebook)
		self.in_ipynb = check_ipynb().is_inipynb()

	def plot_accuracy_history(self, df, save_path):
		# Select only useful columns
		acc_df = df[['sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		# Rename columns to clearer names
		acc_df.rename(inplace=True, index=str, columns={'sparse_top30_accuracy':'Train top 30 accuracy', 'val_sparse_top30_accuracy':'Val top 30 accuracy', 'sparse_top10_accuracy':'Train top 10 accuracy', 'val_sparse_top10_accuracy':'Val top 10 accuracy', 'sparse_categorical_accuracy':'Train top 1 accuracy', 'val_sparse_categorical_accuracy':'Val top 1 accuracy'})
		# Structure the dataframe as expected by Seaborn
		acc_df = acc_df.stack().reset_index()
		acc_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		# Make sure the epochs are int to avoid weird ordering effects in the plot
		acc_df['Epoch'] = acc_df['Epoch'].astype('int8')
		# Plot
		sns.set(style='darkgrid')
		sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=acc_df)
		# Output the plot
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'acc_history.png'))
		# Clear
		plt.gcf().clear()

	def plot_loss_history(self, df, save_path):
		# Select only useful columns
		loss_df = df[['loss', 'val_loss']].copy()
		# Rename columns to clearer names
		loss_df.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		# Structure the dataframe as expected by Seaborn
		loss_df = loss_df.stack().reset_index()
		loss_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		# Make sure the epochs are int to avoid weird ordering effects in the plot
		loss_df['Epoch'] = loss_df['Epoch'].astype('int8')
		# Plot
		sns.set(style='darkgrid')
		sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=loss_df)
		# Output the plot
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'loss_history.png'))
		# Clear
		plt.gcf().clear()

	def plot_crossval_accuracy_history(self, df, save_path):
		# Select only useful columns
		cv_results_df_filtered = df[['sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		# Rename columns to clearer names
		cv_results_df_filtered.rename(inplace=True, index=str, columns={'sparse_top30_accuracy':'Train top 30 accuracy', 'val_sparse_top30_accuracy':'Val top 30 accuracy', 'sparse_top10_accuracy':'Train top 10 accuracy', 'val_sparse_top10_accuracy':'Val top 10 accuracy', 'sparse_categorical_accuracy':'Train top 1 accuracy', 'val_sparse_categorical_accuracy':'Val top 1 accuracy'})
		# Structure the dataframe as expected by Seaborn
		cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
		cv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Split', 'level_1':'Metric', 0:'Result'})
		# Make sure the splits are int to avoid weird ordering effects in the plot
		cv_results_graph_df['Split'] = cv_results_graph_df['Split'].astype('int8')
		# Plot
		sns.set(style='darkgrid')
		sns.relplot(x='Split', y='Result', hue='Metric', kind='line', data=cv_results_graph_df)
		# Output the plot
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'cross_val_acc_history.png'))
		# Clear
		plt.gcf().clear()
	
	def plot_crossval_loss_history(self, df, save_path):
		# Select only useful columns
		cv_results_df_filtered = df[['loss', 'val_loss']].copy()
		# Rename columns to clearer names
		cv_results_df_filtered.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		# Structure the dataframe as expected by Seaborn
		cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
		cv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Split', 'level_1':'Metric', 0:'Result'})
		# Make sure the splits are int to avoid weird ordering effects in the plot
		cv_results_graph_df['Split'] = cv_results_graph_df['Split'].astype('int8')
		# Plot
		sns.set(style='darkgrid')
		sns.relplot(x='Split', y='Result', hue='Metric', kind='line', data=cv_results_graph_df)
		# Output the plot
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'cross_val_loss_history.png'))
		# Clear
		plt.gcf().clear()
