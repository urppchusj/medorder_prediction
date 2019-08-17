# ==============================================================================
# LICENSE GOES HERE
# ==============================================================================

'''
Author: Maxime Thibault

Try to predict the next medication added to the profile
'''

import argparse as ap
import logging
import os
import pathlib
import pickle
import random
import warnings
from datetime import datetime
from multiprocessing import cpu_count

import gensim.utils as gsu
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from gensim.matutils import Sparse2Corpus
from gensim.models import KeyedVectors
from gensim.sklearn_api import LsiTransformer, W2VTransformer
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelBinarizer,
                                   LabelEncoder, label_binarize)

CSVLogger = tf.keras.callbacks.CSVLogger
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
TensorBoard = tf.keras.callbacks.TensorBoard
LSTM = tf.keras.layers.CuDNNLSTM #CuDNNLSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2
BatchNormalization = tf.keras.layers.BatchNormalization
Input = tf.keras.layers.Input
Masking = tf.keras.layers.Masking
concatenate = tf.keras.layers.concatenate
sparse_top_k_categorical_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy
Model = tf.keras.models.Model
load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequence = tf.keras.utils.Sequence

#################
#### CLASSES ####
#################

class data:

	def __init__(self):
		pass

	def get_lists(self, profiles_file, targets_file, seq_file, active_profiles_file, active_classes_file, depa_file, enc_file, restrictdata=False):
		logging.info('Loading data...')
		logging.debug('Loading profiles...')
		with open(profiles_file, mode='rb') as file:
			self.profile_lists = pickle.load(file)
		logging.debug('Loading targets...')
		with open(targets_file, mode='rb') as file:
			self.targets_list = pickle.load(file)
		logging.debug('Loading sequences...')
		with open(seq_file, mode='rb') as file:
			self.seq_lists = pickle.load(file)
		logging.debug('Loading active profiles...')
		with open(active_profiles_file, mode='rb') as file:
			self.active_profile_lists = pickle.load(file)
		logging.debug('Loading active classes...')
		with open(active_classes_file, mode='rb') as file:
			self.active_classes_list = pickle.load(file) 
		logging.debug('Loading depas...')
		with open(depa_file, mode='rb') as file:
			self.depa_list = pickle.load(file) 
		logging.debug('Loading encs...')
		with open(enc_file, mode='rb') as file:
			self.enc_list = pickle.load(file)
		logging.debug('Returning profiles...')
		if restrictdata:
			logging.warning('Data restriction flag enabled, sampling 1000 encounters...')
			sample_size = 1000
			self.enc_list = [self.enc_list[i] for i in sorted(random.sample(range(len(self.enc_list)), sample_size))]
		return self.profile_lists, self.targets_list, self.seq_lists, self.active_profile_lists, self.active_classes_list, self.depa_list, self.enc_list

	def make_lists(self, profiles, targets, seqs, active_profiles, active_classes, depas, enc_train, enc_test, test_type='entire'):
		logging.info('Building data lists...')

		profiles_train = [profiles[enc] for enc in enc_train]
		targets_train = [target for enc in enc_train for target in targets[enc]]
		seq_train = [seq for enc in enc_train for seq in seqs[enc]]
		active_profiles_train = [active_profile for enc in enc_train for active_profile in active_profiles[enc]]
		active_classes_train = [active_class for enc in enc_train for active_class in active_classes[enc]]
		depa_train = [str(depa) for enc in enc_train for depa in depas[enc]]

		unique_targets_train = list(set(targets_train))

		if test_type == 'none':
			targets_test = None
			seq_test = None
			active_profiles_test = None
			active_classes_test = None
			depa_test = None
		else:
			targets_test = [target for enc in enc_test for target in targets[enc] if target in unique_targets_train]
			seq_test = [seq for enc in enc_test for seq, target in zip(seqs[enc], targets[enc]) if target in unique_targets_train]
			active_profiles_test = [active_profile for enc in enc_test for active_profile, target in zip(active_profiles[enc], targets[enc]) if target in unique_targets_train]
			active_classes_test = [active_class for enc in enc_test for active_class, target in zip(active_classes[enc], targets[enc]) if target in unique_targets_train]
			depa_test = [str(depa) for enc in enc_test for depa, target in zip(depas[enc], targets[enc]) if target in unique_targets_train]

		shuffled = list(zip(targets_train, seq_train, active_profiles_train, active_classes_train, depa_train))
		random.shuffle(shuffled)
		targets_train, seq_train, active_profiles_train, active_classes_train, depa_train = zip(*shuffled)

		logging.info('Training set: Obtained {} profiles, {} targets, {} sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(profiles_train), len(targets_train), len(seq_train), len(active_profiles_train), len(active_classes_train), len(depa_train), len(enc_train)))
		
		logging.info('Validation set: Obtained {} targets, {} sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(targets_test), len(seq_test), len(active_profiles_test), len(active_classes_test), len(depa_test), len(enc_test)))

		return profiles_train, targets_train, targets_test, seq_train, seq_test, active_profiles_train, active_profiles_test, active_classes_train, active_classes_test, depa_train, depa_test

	def get_definitions(self):

		definitions_col_names = ['medinb', 'mediname', 'genenb', 'genename', 'classnb', 'classname']
		definitions_dtypes = {'medinb':str, 'mediname':str, 'genenb':str, 'genename':str, 'classnb':str, 'classename':str}
		classes_data = pd.read_csv('data/definitions.csv', sep=';', names=definitions_col_names, dtype=definitions_dtypes)
		definitions = dict(zip(list(classes_data.medinb), list(classes_data.mediname)))

		return definitions


class skl_model:

	def __init__(self):
		self.skl_save_path = os.path.join(os.getcwd(), 'model', 'optimized')
		pathlib.Path(self.skl_save_path).mkdir(parents=True, exist_ok=True)

	def w2v_processor(self, w2v_embedding_dim):
		logging.debug('Building word2vec pipeline...')
		pipe = Pipeline([
			('w2v', W2VTransformer(alpha=0.013, iter=32, size=w2v_embedding_dim, hs=0, sg=0, min_count=5, workers=cpu_count()))
			# for 10 years (also change w2v_embedding_dim) (alpha=0.007, iter=64, size=w2v_embedding_dim, hs=0, sg=0, min_count=5, workers=cpu_count())),
			# for 5 years (alpha=0.013, iter=32, size=w2v_embedding_dim, hs=0, sg=0, min_count=5, workers=cpu_count()))
		])
		return pipe
	
	def profile_state_encoder(self, n_col, use_lsi, tsvd_n_components):
		logging.debug('Building profile state encoder...')
		logging.debug('Building ColumnTransformer of Count Vectorizers for {} pipes...'.format(n_col))
		transformers = []
		for i in range(n_col):
			transformers.append(('pse{}'.format(i), CountVectorizer(lowercase=False, preprocessor=self.pse_pp, analyzer=self.pse_a), i))
		pipeline_transformers = [
			('columntrans', ColumnTransformer(transformers=transformers))
			]
		if use_lsi == True:
			pipeline_transformers.extend([
				('tfidf', TfidfTransformer()),
				('sparse2corpus', FunctionTransformer(func=Sparse2Corpus, accept_sparse=True, validate=False, kw_args={'documents_columns':False})),
				('tsvd', LsiTransformer(tsvd_n_components))
			])
		pipe = Pipeline(pipeline_transformers)
		return pipe
	
	def pse_pp(self, x):
		return ' '.join(x)

	def pse_a(self, x):
		return x

	def target_encoder(self):
		logging.debug('Building label encoder for target...')
		le = LabelEncoder()
		return le

	def target_binarizer(self):
		logging.debug('Building label binarizer for target...')
		lb = LabelBinarizer()
		return lb

	def fit_pipelines(self, profiles, targets, active_profiles, active_classes, depa, w2v_embedding_dim, use_lsi, tsvd_n_components, export_w2v_embeddings=False):
		logging.debug('Building and fitting skl pipelines...')
		
		logging.debug('Fitting word2vec...')
		w2v = self.w2v_processor(w2v_embedding_dim)

		w2v.fit(profiles)
		w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)
		if export_w2v_embeddings:
			self.export_w2v_embeddings(w2v)

		logging.debug('Computing data for LSI...')
		pse_data = [[ap, ac, de] for ap, ac, de in zip(active_profiles, active_classes, depa)]
		logging.debug('Fitting LSI...')
		pse = self.profile_state_encoder(len(pse_data[0]), use_lsi, tsvd_n_components)
		pse.fit(pse_data)

		logging.debug('Fitting label encoder')
		le = self.target_encoder()
		le.fit(targets)

		logging.debug('Fitting label binarizer')
		lb = self.target_binarizer()
		lb.fit(targets)
		
		return w2v, pse, le, lb

	def export_w2v_embeddings(self, w2v):
		w2v.named_steps['w2v'].gensim_model.wv.save_word2vec_format(os.path.join(self.skl_save_path, save_timestamp + 'w2v.model'))
		model = KeyedVectors.load_word2vec_format(os.path.join(self.skl_save_path, save_timestamp + 'w2v.model'), binary=False)
		outfiletsv = os.path.join(self.skl_save_path, save_timestamp + '_tensor.tsv')
		outfiletsvmeta = os.path.join(self.skl_save_path, save_timestamp + '_metadata.tsv')

		with open(outfiletsv, 'w+') as file_vector:
			with open(outfiletsvmeta, 'w+') as file_metadata:
				gen = (word for word in model.index2word if word != 'unk')
				for word in gen:
					file_metadata.write(gsu.to_utf8(word).decode('utf-8') + gsu.to_utf8('\n').decode('utf-8'))
					vector_row = '\t'.join(str(x) for x in model[word])
					file_vector.write(vector_row + '\n')

		logging.info("2D tensor file saved to %s", outfiletsv)
		logging.info("Tensor metadata file saved to %s", outfiletsvmeta)

		definitions = data().get_definitions()

		with open(os.path.join(outfiletsvmeta), mode='r', encoding='utf-8', errors='strict') as metadata_file:
			metadata = metadata_file.read()
		converted_string = ''
		for element in metadata.splitlines():
			string = element.strip()
			converted_string +=	definitions[string] + '\n'
		with open(os.path.join(self.skl_save_path, save_timestamp + '_defined_metadata.tsv'), mode='w', encoding='utf-8', errors='strict') as converted_metadata:
			converted_metadata.write(converted_string)


class keras_model:

	def __init__(self):
		self.keras_save_path = os.path.join(os.getcwd(), 'model', 'optimized')
		pathlib.Path(self.keras_save_path).mkdir(parents=True, exist_ok=True)

	def define_model(self, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, tsvd_n_components, output_n_classes):
		logging.debug('Building model...')

		to_concat = []
		inputs= []
		
		w2v_input = Input(shape=(sequence_length, w2v_embedding_dim, ), dtype='float32', name='w2v_input')
		w2v = LSTM(lstm_size, return_sequences=True)(w2v_input)
		w2v = Dropout(dropout)(w2v)
		w2v = LSTM(lstm_size)(w2v)
		w2v = Dropout(dropout)(w2v)
		w2v = Dense(lstm_size, activation='relu', kernel_regularizer=l2(l2_reg))(w2v)
		w2v = Dropout(dropout)(w2v)
		to_concat.append(w2v)
		inputs.append(w2v_input)
		
		pse_input = Input(shape=(tsvd_n_components,), dtype='float32', name='pse_input')
		pse = Dense(dense_pse_size, activation='relu', kernel_regularizer=l2(l2_reg))(pse_input)
		pse = Dropout(dropout)(pse)
		to_concat.append(pse)
		inputs.append(pse_input)

		concatenated = concatenate(to_concat)
		concatenated = BatchNormalization()(concatenated)
		concatenated = Dense(concat_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated)
		concatenated = Dropout(dropout*1.5)(concatenated)
		concatenated = BatchNormalization()(concatenated)
		concatenated = Dense(dense_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated)
		concatenated = Dropout(dropout)(concatenated)
		concatenated = BatchNormalization()(concatenated)
		output = Dense(output_n_classes, activation='softmax', name='main_output')(concatenated)

		model = Model(inputs = inputs, outputs = output)
		model.compile(optimizer='Adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy', self.sparse_top10_accuracy, self.sparse_top30_accuracy])
		print(model.summary())
		
		return model

	def single_pass(self, save_stamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size, epochs=1000):
		logging.info('Splitting encounters into train and test sets...')
		enc_train, enc_test = train_test_split(enc, shuffle=False, test_size=0.25)
		d = data()
		profiles_train, targets_train, targets_test, seq_train, seq_test, active_profiles_train, active_profiles_test, active_classes_train, active_classes_test, depa_train, depa_test = d.make_lists(profiles, targets, seqs, active_profiles, active_classes, depa, enc_train, enc_test)
		s = skl_model()
		w2v, pse, le, _ = s.fit_pipelines(profiles_train, targets_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, use_lsi, tsvd_n_components, export_w2v_embeddings=True)
		w2v_step = w2v.named_steps['w2v']
		if use_lsi == False:
			tsvd_n_components = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
		output_n_classes = len(le.classes_)
		train_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_train, seq_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, sequence_length, batch_size)
		test_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_test, seq_test, active_profiles_test, active_classes_test, depa_test, w2v_embedding_dim, sequence_length, batch_size, shuffle=False)
		logging.info('Training keras model...')
		rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.0005)
		tb_callback = TensorBoard(os.path.join(os.getcwd(), 'logs', 'build_model', 'keras', 'optimized', save_timestamp))
		earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, restore_best_weights=True)
		model = self.define_model(lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, tsvd_n_components, output_n_classes)
		tf.keras.utils.plot_model(model, to_file = os.path.join(os.path.join('model', 'optimized', save_stamp + '_model.png')))
		history = model.fit_generator(train_generator,
			epochs=epochs,
			callbacks=[tb_callback, rlr_callback, earlystop_callback],
			validation_data=test_generator)
		history_df = pd.DataFrame(history.history)
		acc_df = history_df[['sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		acc_df.rename(inplace=True, index=str, columns={'sparse_top10_accuracy':'Train top 10 accuracy', 'val_sparse_top10_accuracy':'Val top 10 accuracy', 'sparse_top30_accuracy':'Train top 30 accuracy', 'val_sparse_top30_accuracy':'Val top 30 accuracy', 'sparse_categorical_accuracy':'Train top 1 accuracy', 'val_sparse_categorical_accuracy':'Val top 1 accuracy'})
		acc_df = acc_df.stack().reset_index()
		acc_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		acc_df['Epoch'] = acc_df['Epoch'].astype('int8')
		sns.set(style='darkgrid')
		f = sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=acc_df)
		plt.savefig(os.path.join('model', 'optimized', save_stamp + '_acc_history.png'))
		plt.gcf().clear()
		loss_df = history_df[['loss', 'val_loss']].copy()
		loss_df.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		loss_df = loss_df.stack().reset_index()
		loss_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		loss_df['Epoch'] = loss_df['Epoch'].astype('int8')
		sns.set(style='darkgrid')
		f = sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=loss_df)
		plt.savefig(os.path.join('model', 'optimized', save_stamp + '_loss_history.png'))

	def single_cross_val_internal(self, save_stamp, tb_contextlist, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size, epochs=1000):
		sizes_list = []
		to_concat = []
		i=0
		for train, test in TimeSeriesSplit(n_splits=5).split(enc):
			enc_train = [enc[i] for i in train]
			enc_test = [enc[i] for i in test]
			logging.info('Splitting encounters into train and test sets...')
			d = data()
			profiles_train, targets_train, targets_test, seq_train, seq_test, active_profiles_train, active_profiles_test, active_classes_train, active_classes_test, depa_train, depa_test = d.make_lists(profiles, targets, seqs, active_profiles, active_classes, depa, enc_train, enc_test)
			s = skl_model()
			w2v, pse, le, _ = s.fit_pipelines(profiles_train, targets_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, use_lsi, tsvd_n_components)
			w2v_step = w2v.named_steps['w2v']
			if use_lsi == False:
				tsvd_n_components = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
			output_n_classes = len(le.classes_)
			train_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_train, seq_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, sequence_length, batch_size)
			test_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_test, seq_test, active_profiles_test, active_classes_test, depa_test, w2v_embedding_dim, sequence_length, batch_size, shuffle=False)
			logging.info('Training keras model...')
			rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.0005)
			if tb_contextlist[0] == 'scv':
				tb_loc = os.path.join(os.getcwd(), 'logs', 'build_model', 'keras', 'optimized', save_timestamp, 'scv_step_{}'.format(i))
			elif tb_contextlist[0] == 'lc':
				tb_loc = os.path.join(os.getcwd(), 'logs', 'build_model', 'keras', 'optimized', save_timestamp, 'lc_step_{}'.format(tb_contextlist[1]), 'cv_step_{}'.format(i))
			tb_callback = TensorBoard(tb_loc)
			earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, restore_best_weights=True)
			csv_callback = CSVLogger(os.path.join(self.keras_save_path, save_timestamp + '_single_cross_val_history.csv'), append=True)
			model = self.define_model(lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, tsvd_n_components, output_n_classes)
			history = model.fit_generator(train_generator,
				epochs=epochs,
				callbacks=[tb_callback, rlr_callback, earlystop_callback, csv_callback],
				validation_data=test_generator)
			fold_results_df = pd.DataFrame(history.history)
			fold_results_df['Split'] = i
			i += 1
			to_concat.append(fold_results_df.tail(1))
			sizes_list.append({'train':len(seq_train), 'test':len(seq_test)})
		cv_results_df = pd.concat(to_concat)
		return cv_results_df, sizes_list

	def single_cross_val_model(self, save_stamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size):
		logging.debug('Performing single cross validation of keras model...')
		cv_results_df, sizes_list = self.single_cross_val_internal(save_stamp, ['scv'], lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size)
		cv_results_df_filtered = cv_results_df[['Split', 'sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		cv_results_df_filtered.rename(inplace=True, index=str, columns={'sparse_top10_accuracy':'Train top 10 accuracy', 'val_sparse_top10_accuracy':'Val top 10 accuracy', 'sparse_top30_accuracy':'Train top 30 accuracy', 'val_sparse_top30_accuracy':'Val top 30 accuracy', 'sparse_categorical_accuracy':'Train top 1 accuracy', 'val_sparse_categorical_accuracy':'Val top 1 accuracy'})
		cv_results_df_filtered.set_index('Split', inplace=True)
		cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
		cv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Split', 'level_1':'Metric', 0:'Result'})
		sns.set(style='darkgrid')
		f = sns.relplot(x='Split', y='Result', hue='Metric', kind='line', data=cv_results_graph_df)
		f.set(xticks=np.arange(0,len(sizes_list),1))
		plt.savefig(os.path.join('model', 'optimized', save_stamp + '_single_cross_val_accuracy_results.png'))
		plt.gcf().clear()
		cv_results_df_filtered = cv_results_df[['Split', 'loss', 'val_loss']].copy()
		cv_results_df_filtered.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		cv_results_df_filtered.set_index('Split', inplace=True)
		cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
		cv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Split', 'level_1':'Metric', 0:'Result'})
		sns.set(style='darkgrid')
		f = sns.relplot(x='Split', y='Result', hue='Metric', kind='line', data=cv_results_graph_df)
		f.set(xticks=np.arange(0,len(sizes_list),1))
		plt.savefig(os.path.join('model', 'optimized', save_stamp + '_single_cross_val_accuracy_loss.png'))

	def learning_curve(self, save_stamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size):
		logging.debug('Performing learning curve of keras model')
		to_concat = []
		j=0
		for boundary in np.linspace(0, len(profiles), 6):
			if boundary == 0:
				continue
			save_timestamp = save_stamp + '_step{}_'.format(j)
			step_enc = enc[:int(boundary)]
			step_results_df, sizes_list = self.single_cross_val_internal(save_stamp, ['lc', j], lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, step_enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size)
			step_results_df['Training examples'] = max([size_dict['train'] for size_dict in sizes_list])
			to_concat.append(step_results_df)
			j += 1
		learning_curve_results = pd.concat(to_concat)
		learning_curve_filtered = learning_curve_results[['Training examples', 'sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		learning_curve_filtered.rename(inplace=True, index=str, columns={'sparse_top10_accuracy':'Train top 10', 'val_sparse_top10_accuracy':'Val top 10', 'sparse_top30_accuracy':'Train top 30', 'val_sparse_top30_accuracy':'Val top 30', 'sparse_categorical_accuracy':'Train top 1', 'val_sparse_categorical_accuracy':'Val top 1'})
		learning_curve_filtered.set_index('Training examples', inplace=True)
		learning_curve_graph_df = learning_curve_filtered.stack().reset_index()
		learning_curve_graph_df.rename(inplace=True, index=str, columns={'level_1':'Metric', 0:'Accuracy'})
		print(learning_curve_graph_df.head())
		sns.set(style='darkgrid')
		sns.relplot(x='Training examples', y='Accuracy', hue='Metric', kind='line', data=learning_curve_graph_df, hue_order=['Train top 30', 'Val top 30', 'Train top 10', 'Val top 10', 'Train top 1', 'Val top 1'])
		plt.savefig(os.path.join('model', 'optimized', save_timestamp + 'learning_curve_accuracy.png'))
		plt.gcf().clear()
		learning_curve_filtered = learning_curve_results[['Training examples', 'loss', 'val_loss']].copy()
		learning_curve_filtered.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		learning_curve_filtered.set_index('Training examples', inplace=True)
		learning_curve_graph_df = learning_curve_filtered.stack().reset_index()
		learning_curve_graph_df.rename(inplace=True, index=str, columns={'level_1':'Metric', 0:'Accuracy'})
		print(learning_curve_graph_df.head())
		sns.set(style='darkgrid')
		sns.relplot(x='Training examples', y='Accuracy', hue='Metric', kind='line', data=learning_curve_graph_df, hue_order=['Train loss', 'Val loss'])
		plt.savefig(os.path.join('model', 'optimized', save_timestamp + 'learning_curve_loss.png'))
		

	def single_train_save(self, save_stamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size, epochs):
		s = skl_model()
		profiles_train, targets_train, _, seq_train, _, active_profiles_train, _, active_classes_train, _, depa_train, _ = data().make_lists(profiles, targets, seqs, active_profiles, active_classes, depa, enc, None, test_type='none')
		w2v, pse, le, lb = s.fit_pipelines(profiles_train, targets_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, use_lsi, tsvd_n_components, export_w2v_embeddings=True)
		dump([w2v, pse, le, lb], os.path.join(self.keras_save_path, save_timestamp + 'sklpipe.joblib'))
		w2v_step = w2v.named_steps['w2v']
		if use_lsi == False: 
			tsvd_n_components = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
		output_n_classes = len(le.classes_)
		train_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_train, seq_train, active_profiles_train, active_classes_train, depa_train, w2v_embedding_dim, sequence_length, batch_size)
		logging.info('Training keras model...')
		tb_callback = TensorBoard(os.path.join(os.getcwd(), 'logs', 'build_model', 'keras', 'optimized', save_timestamp))
		model = self.define_model(lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, tsvd_n_components, output_n_classes)
		model.fit_generator(train_generator,
			epochs=epochs,
			callbacks=[tb_callback])
		logging.info('Saving model...')
		model.save(os.path.join(self.keras_save_path, save_timestamp + 'keras_model.h5'))

	def eval(self, save_stamp, sklfile, kerasfile, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, use_lsi, batch_size):
		w2v, pse, le, _ = load(sklfile)
		w2v_step = w2v.named_steps['w2v']
		_, targets_eval, _, seq_eval, _, active_profiles_eval, _, active_classes_eval, _, depa_eval, _ = data().make_lists(profiles, targets, seqs, active_profiles, active_classes, depa, enc, None, test_type='none')
		pre_discard_n_targets = len(set(targets_eval))
		targets_eval = [target for target in targets_eval if target in w2v_step.gensim_model.wv.index2entity]
		seq_eval = [seq for seq, target in zip(seq_eval, targets_eval) if target in w2v_step.gensim_model.wv.index2entity]
		active_profiles_eval = [active_profile for active_profile, target in zip(active_profiles_eval, targets_eval) if target in w2v_step.gensim_model.wv.index2entity]
		active_classes_eval = [active_class for active_class, target in zip(active_classes_eval, targets_eval) if target in w2v_step.gensim_model.wv.index2entity]
		depa_eval = [depa for depa, target in zip(depa_eval, targets_eval) if target in w2v_step.gensim_model.wv.index2entity]
		post_discard_n_targets = len(set(targets_eval))
		eval_generator = TransformedGenerator(w2v_step, use_lsi, pse, le, targets_eval, seq_eval, active_profiles_eval, active_classes_eval, depa_eval, w2v_embedding_dim, sequence_length, batch_size)
		logging.info('Evaluating keras model...')
		model = load_model(kerasfile, custom_objects={'sparse_top10_accuracy':self.sparse_top10_accuracy, 'sparse_top30_accuracy':self.sparse_top30_accuracy})
		results = model.evaluate_generator(eval_generator, verbose=1)
		predictions = model.predict_generator(eval_generator, verbose=1)
		logging.info('Evaluation results')
		logging.info('Predicting on {} samples, {:.2f} % of {} samples, {} samples discarded because of unseen labels.'.format(len(targets_eval), 100*post_discard_n_targets/pre_discard_n_targets, pre_discard_n_targets, pre_discard_n_targets-post_discard_n_targets))
		logging.info('Predictions for {} classes'.format(len(le.classes_)))
		logging.info('{} classes reprensented in targets'.format(len(set(targets_eval))))
		for metric, result in zip(model.metrics_names, results):
			logging.info('Metric: {}   Score: {:.5f}'.format(metric,result))
		prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
		pl_series = pd.Series(prediction_labels)
		f = sns.countplot(pl_series, order=pl_series.value_counts().index[:50])
		f.set(xticklabels='', xlabel='classes')
		plt.savefig('predictions_distribution.png')
		nb_predicted_classes = len(pl_series.value_counts().index)
		print('Number of classes predicted on evaluation set: {}'.format(nb_predicted_classes))
		cr = classification_report(targets_eval, prediction_labels, output_dict=True)
		cr_df = pd.DataFrame.from_dict(cr, orient='index')
		cr_df.to_csv(os.path.join(self.keras_save_path, save_timestamp + 'eval_classification_report.csv'))
		binary_labels = label_binarize(targets_eval, le.classes_)
		filtered_binary_labels = np.delete(binary_labels,np.where(~binary_labels.any(axis=0))[0], axis=1)
		filtered_predictions = np.delete(predictions,np.where(~binary_labels.any(axis=0))[0], axis=1)
		rocauc_ma = roc_auc_score(filtered_binary_labels, filtered_predictions, average='macro')
		rocauc_mi = roc_auc_score(filtered_binary_labels, filtered_predictions, average='micro')
		rocauc_we = roc_auc_score(filtered_binary_labels, filtered_predictions, average='weighted')
		logging.info('Macro average ROC AUC score for present labels: {:.3f}'.format(rocauc_ma))
		logging.info('Micro average ROC AUC score for present labels: {:.3f}'.format(rocauc_mi))
		logging.info('Weighted average ROC AUC score for present labels: {:.3f}'.format(rocauc_we))
		logging.info(cr_df.describe())

	def sparse_top10_accuracy(self, y_true, y_pred):
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=10))

	def sparse_top30_accuracy(self, y_true, y_pred):
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=30))


class TransformedGenerator(Sequence):

	def __init__(self, w2v, use_lsi, pse, le, y, X_w2v, X_ap, X_ac, X_depa, w2v_embedding_dim, sequence_length, batch_size, shuffle=True, return_targets=True):
		self.w2v = w2v
		self.use_lsi = use_lsi
		self.pse = pse
		self.le = le
		self.y = y
		self.X_w2v = X_w2v
		self.X_ap = X_ap
		self.X_ac = X_ac
		self.X_depa = X_depa
		self.w2v_embedding_dim = w2v_embedding_dim
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.return_targets = return_targets

	def __len__(self):
		return int(np.ceil(len(self.X_w2v) / float(self.batch_size)))

	def __getitem__(self, idx):
		X = dict()
		batch_w2v = self.X_w2v[idx * self.batch_size:(idx+1) * self.batch_size]
		transformed_w2v = [[self.w2v.gensim_model.wv.get_vector(medic) for medic in seq if medic in self.w2v.gensim_model.wv.index2entity] for seq in batch_w2v]
		transformed_w2v = [l if len(l) > 0 else [np.zeros(self.w2v_embedding_dim)] for l in transformed_w2v]
		transformed_w2v = pad_sequences(transformed_w2v, maxlen=self.sequence_length, dtype='float32')
		X['w2v_input']=transformed_w2v
		batch_ap = self.X_ap[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_ac = self.X_ac[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_depa = self.X_depa[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_pse = [[bp, bc, bd] for bp, bc, bd in zip(batch_ap, batch_ac, batch_depa)]
		transformed_pse = self.pse.transform(batch_pse)
		X['pse_input']=transformed_pse
		if self.use_lsi == False:
			X['pse_input']=X['pse_input'].todense()
		if self.return_targets:
			batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
			transformed_y = self.le.transform(batch_y)
			y = {'main_output': transformed_y}
			return X, y
		else:
			return X
	
	def on_epoch_end(self):
		if self.shuffle == True:
			shuffled = list(zip(self.y, self.X_w2v, self.X_ap, self.X_ac, self.X_depa))
			random.shuffle(shuffled)
			self.y, self.X_w2v, self.X_ap, self.X_ac, self.X_depa = zip(*shuffled)


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	parser = ap.ArgumentParser(description='Build and evaluate an autoencoder for pharmacological profiles to obtain reconstruction losses for use in an anomaly detection algorithm.', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--datadir', metavar='Type_String', type=str, nargs="?", default='preprocessed_data/optimized/', help='Directory where preprocessed data is stored. Defaults to "preprocessed_data/optimized/ if no argument is specified. If specified, must be a subdirectory of this directory.')
	parser.add_argument('--sklfile', metavar='Type_String', type=str, nargs="?", default='', help='Trained skl file to use with eval mode. No default.')
	parser.add_argument('--kerasfile', metavar='Type_String', type=str, nargs="?", default='', help='Trained keras file to use with eval mode. No default.')
	parser.add_argument('--op', metavar='Type_String', type=str, nargs="?", default='scv', help='Use "sp" to perform a single pass over all the data and log the results to TensorBoard. Use "scv" to perform a single cross-validation and scoring results of splits. Use "lc" to plot a learning curve. Use "sts" to perform a single train and save the model. Use "eval" to perform evaluation on trained model; specify --sklfile and --kerasfile locations. Defaults to scv.')
	parser.add_argument('--verbose', action='store_true', help='Use this argument to log at DEBUG level, otherwise logging will occur at level INFO.')
	parser.add_argument('--restrictdata', action='store_true', help='Use this argument to restrict the number of encounters to 1000 (randomly sampled but keeping ordering) to reduce execution time during debugging.')

	args = parser.parse_args()
	data_dir = args.datadir
	skl_file = args.sklfile
	keras_file = args.kerasfile
	op = args.op
	verbose = args.verbose
	restrict_data = args.restrictdata

	# check arguments
	if op not in ['sp', 'scv', 'lc', 'sts', 'eval']:
		logging.critical('Operation {} not implemented. Quitting...'.format(op))
		quit()
	
	save_timestamp = datetime.now().strftime('%Y%m%d-%H%M') + op

	#disable user warnings (otherwise unseen classes for test data generate warnings during cross validation)
	warnings.simplefilter('ignore', category=UserWarning)
	# Logger
	print('Configuring logger...')
	if verbose == True :
		ll = logging.DEBUG
	else:
		ll = logging.INFO
	logging_path = os.path.join(os.getcwd(), 'logs', 'build_model', 'keras', 'optimized', save_timestamp)
	pathlib.Path(logging_path).mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=ll,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.FileHandler(os.path.join(logging_path, save_timestamp + '.log')),
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')

	if restrict_data:
		logging.warning('Data restriction flag enabled !')

	logging.info('Performing operation: {}'.format(op))
	
	# Get the data
	logging.debug('Obtaining data...')
	profiles_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'profiles_list.pkl')
	targets_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'targets_list.pkl')
	seq_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'seq_list.pkl')
	activeprofiles_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'active_profiles_list.pkl')
	activeclasses_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'active_classes_list.pkl')
	depa_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'depa_list.pkl')
	enc_path = os.path.join(os.getcwd(), 'preprocessed_data', 'optimized', data_dir, 'enc_list.pkl')
	profiles, targets, seqs, active_profiles, active_classes, depa, enc = data().get_lists(profiles_path, targets_path, seq_path, activeprofiles_path, activeclasses_path, depa_path, enc_path, restrict_data)

	lstm_size = 128 # baseline 128 (64, 128, 256)
	dense_pse_size = 128 # baseline 64 (32, 64, 128)
	concat_size = 256 # baseline 128 (64, 128, 256)
	dense_size = 256
	dropout = 0.2 # 0.2 best accuracy, 0.5 less overfitting, 0.4 accuracy converges
	l2_reg = 0
	w2v_embedding_dim = 128 # baseline 128 (for 10 years use 64)
	sequence_length = 30 # baseline 30
	batch_size = 256
	single_train_epochs = 20
	use_lsi = False
	if restrict_data:
		tsvd_n_components = 200
	else:
		tsvd_n_components = 500 # baseline 500
	

	# Execute
	k = keras_model()
	
	if op == 'sp':
		k.single_pass(save_timestamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size)
	if op == 'scv':
		k.single_cross_val_model(save_timestamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size)
	elif op == 'lc':
		k.learning_curve(save_timestamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size)
	elif op == 'sts':
		k.single_train_save(save_timestamp, lstm_size, dense_pse_size, concat_size, dense_size, dropout, l2_reg, profiles, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, w2v_embedding_dim, use_lsi, tsvd_n_components, batch_size, single_train_epochs)
	elif op == 'eval':
		k.eval(save_timestamp, skl_file, keras_file, targets, seqs, active_profiles, active_classes, depa, enc, sequence_length, use_lsi, batch_size)
	quit()
