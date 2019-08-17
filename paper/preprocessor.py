import argparse as ap
import logging
import os
import pathlib
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd

###########
# CLASSES #
###########

class preprocessor():

	def __init__(self, source_file, definitions_file, restrict_data, logging_level=logging.DEBUG):

		# Where the preprocessed files will be saved
		self.data_save_path = os.path.join(os.getcwd(), 'paper', 'preprocessed_data')
		# Column names in the data file
		self.profile_col_names = ['enc', 'date_beg', 'time_beg', 'date_end', 'time_end', 'medinb', 'date_begenc', 'date_endenc', 'time_endenc', 'depa']
		# dtypes of data file columns
		self.profile_dtypes = {'enc':np.int32, 'date_beg':str, 'time_beg':str, 'date_end':str, 'time_end':str, 'medinb':str, 'date_begenc':str, 'date_endenc':str, 'time_endenc':str, 'depa':str}
		# Column names in the definitions file
		self.definitions_col_names = ['medinb', 'mediname', 'classnb', 'classname']
		# dtypes of definitions files columns
		self.definitions_dtypes = {'medinb':np.int32, 'mediname':str, 'classnb':str, 'classename':str}

		# Congigure logger
		print('Configuring logger...')
		self.logging_path = os.path.join(os.getcwd(), 'paper', 'logs', 'preprocessing')
		pathlib.Path(self.logging_path).mkdir(parents=True, exist_ok=True)
		logging.basicConfig(
			level=logging_level,
			format="%(asctime)s [%(levelname)s]  %(message)s",
			handlers=[
				logging.FileHandler(os.path.join(self.logging_path, datetime.now().strftime('%Y%m%d-%H%M') + '.log')),
				logging.StreamHandler()
			])
		logging.debug('Logger successfully configured.')
		
		# Load raw data
		logging.info('Loading data...')
		self.raw_profile_data = pd.read_csv(source_file, names=self.profile_col_names, index_col=None, dtype=self.profile_dtypes)
		classes_data = pd.read_csv(definitions_file, names=self.definitions_col_names, index_col=0, dtype=self.definitions_dtypes)

		# Calculate synthetic features
		'''
		Convert medinb from text to int
		Add classes from the definitions file and decompose into 4 class levels
		Convert dates and times from text to datetime
		Calculate addition numbers which be used later for sequence generation
		Drop data that is not useful anymore (to save RAM)
		'''
		logging.info('Calculating synthetic features...')
		self.raw_profile_data['medinb_int'] = self.raw_profile_data['medinb'].astype(np.int32)
		self.raw_profile_data['classnb'] = self.raw_profile_data['medinb_int'].map(classes_data['classnb'])
		del classes_data
		self.raw_profile_data['class1_part'] = self.raw_profile_data['classnb'].str.slice(start=0, stop=2).astype(np.int32)
		self.raw_profile_data['class2_part'] = self.raw_profile_data['classnb'].str.slice(start=3, stop=5).astype(np.int32)
		self.raw_profile_data['class3_part'] = self.raw_profile_data['classnb'].str.slice(start=6, stop=8).astype(np.int32)
		self.raw_profile_data['class4_part'] = self.raw_profile_data['classnb'].str.slice(start=9, stop=11).astype(np.int32)
		self.raw_profile_data['class1_whole'] = self.raw_profile_data['classnb'].str.slice(start=0, stop=2)
		self.raw_profile_data['class2_whole'] = self.raw_profile_data['classnb'].str.slice(start=0, stop=5)
		self.raw_profile_data['class3_whole'] = self.raw_profile_data['classnb'].str.slice(start=0, stop=8)
		self.raw_profile_data['class4_whole'] = self.raw_profile_data['classnb'].str.slice(start=0, stop=11)
		self.raw_profile_data['datetime_beg'] = pd.to_datetime(self.raw_profile_data['date_beg']+' '+self.raw_profile_data['time_beg'], format='%Y%m%d %H:%M')
		self.raw_profile_data = self.raw_profile_data.drop(['date_beg', 'time_beg'], axis=1)
		self.raw_profile_data['datetime_end'] = pd.to_datetime(self.raw_profile_data['date_end']+' '+self.raw_profile_data['time_end'], format='%Y%m%d %H:%M')
		self.raw_profile_data = self.raw_profile_data.drop(['date_end', 'time_end'], axis=1)
		self.raw_profile_data['date_begenc'] = pd.to_datetime(self.raw_profile_data['date_begenc'], format='%Y%m%d')
		self.raw_profile_data['datetime_endenc'] = pd.to_datetime(self.raw_profile_data['date_endenc']+' '+self.raw_profile_data['time_endenc'], format='%Y%m%d %H:%M')
		self.raw_profile_data = self.raw_profile_data.drop(['date_endenc', 'time_endenc'], axis=1)
		self.raw_profile_data.sort_values(['date_begenc', 'enc', 'datetime_beg', 'class1_part', 'class2_part', 'class3_part', 'class4_part'], ascending=True, inplace=True)
		self.raw_profile_data['addition_number'] = self.raw_profile_data.groupby('enc').enc.rank(method='first').astype(int)
		self.raw_profile_data.set_index(['enc', 'addition_number'], drop=True, inplace=True)
		maxyear = max(self.raw_profile_data['date_begenc'].apply(lambda x: x.year))
		self.raw_profile_data = self.raw_profile_data.loc[self.raw_profile_data['date_begenc'] > datetime(maxyear-int(restrict_data)+1,1,1)].copy()

	def get_profiles(self):
		# Rebuild profiles at every addition
		logging.info('Recreating profiles... (takes a while)')
		profiles_dict = defaultdict(list)
		targets_dict = defaultdict(list)
		seq_dict = defaultdict(list)
		active_profiles_dict = defaultdict(list)
		active_classes_dict = defaultdict(list)
		depa_dict = defaultdict(list)
		enc_list = []
		# Prepare a variable of the number of encounters in the dataset
		length = self.raw_profile_data.index.get_level_values(0).nunique()
		# Iterate over encounters, send each encounter to self.build_enc_profiles
		for n, enc in zip(range(0, length), self.raw_profile_data.groupby(level='enc', sort=False)):
			enc_list.append(enc[0])
			profiles_dict[enc[0]]=enc[1]['medinb'].tolist()
			enc_profiles = self.build_enc_profiles(enc)
			# Convert each profile to list
			for profile in enc_profiles.groupby(level='profile', sort=False):
				print('Handling encounter number {} profile number {}: {:.2f} %\r'.format(enc[0], profile[0], 100*n / length), end='    ', flush=True)
				target, seq_to_append, active_profile_to_append, class_1_to_append, class_2_to_append, class_3_to_append, class_4_to_append, depa_to_append = self.make_profile_lists(profile)
				targets_dict[enc[0]].append(target)
				seq_dict[enc[0]].append(seq_to_append)
				active_profiles_dict[enc[0]].append(active_profile_to_append)
				depa_dict[enc[0]].append(depa_to_append)
				active_classes_dict[enc[0]].append(list(chain.from_iterable([class_1_to_append, class_2_to_append, class_3_to_append, class_4_to_append])))
		logging.info('Done!')
		return profiles_dict, targets_dict, seq_dict, active_profiles_dict, active_classes_dict, depa_dict, enc_list

	def build_enc_profiles(self, enc):
		enc_profiles_list = []
		# Iterate over additions in the encounter
		for addition in enc[1].itertuples():
			# For each addition, generate a profile of all medications with a datetime of beginning
			# before or at the same time of the addition
			profile_at_time = enc[1].loc[(enc[1]['datetime_beg'] <= addition.datetime_beg)].copy()
			# Determine if each medication was active at the time of addition
			profile_at_time['active'] = np.where(profile_at_time['datetime_end'] > addition.datetime_beg, 1, 0)
			# Manipulate indexes to have three levels: encounter, profile and addition
			profile_at_time['profile'] = addition.Index[1]
			profile_at_time.set_index('profile', inplace=True, append=True)
			profile_at_time = profile_at_time.swaplevel(i='profile', j='addition_number')
			enc_profiles_list.append(profile_at_time)
		enc_profiles = pd.concat(enc_profiles_list)
		return enc_profiles

	def make_profile_lists(self, profile):
		# make a list with all medications in profile
		mask = profile[1].index.get_level_values('profile')==profile[1].index.get_level_values('addition_number')
		target = profile[1][mask]['medinb'].astype(str).values[0]
		seq_to_append = profile[1]['medinb'].tolist()
		target_index = len(seq_to_append) - 1 - seq_to_append[::-1].index(target)
		seq_to_append.pop(target_index)
		# remove row of target from profile
		filtered_profile = profile[1].drop(profile[1].index[target_index])
		# select only active medications and make another list with those
		active_profile = filtered_profile.loc[filtered_profile['active'] == 1].copy()
		# make lists of contents of active profile to prepare for multi-hot encoding
		active_profile_to_append = active_profile['medinb'].tolist()
		class_1_to_append = active_profile['class1_whole'].tolist()
		class_2_to_append = active_profile['class2_whole'].tolist()
		class_3_to_append = active_profile['class3_whole'].tolist()
		class_4_to_append = active_profile['class4_whole'].tolist()
		depa_to_append = active_profile['depa'].unique().tolist()
		return target, seq_to_append, active_profile_to_append, class_1_to_append, class_2_to_append, class_3_to_append, class_4_to_append, depa_to_append

	def preprocess(self):
		# Preprocess the data
		profiles_dict, targets_dict, seq_dict, active_profiles_dict, active_classes_dict, depa_dict, enc_list = self.get_profiles()
		# Save preprocessed data to pickle files
		pathlib.Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
		with open(os.path.join(self.data_save_path, 'profiles_list.pkl'), mode='wb') as file:
			pickle.dump(profiles_dict, file)
		with open(os.path.join(self.data_save_path, 'targets_list.pkl'), mode='wb') as file:
			pickle.dump(targets_dict, file)
		with open(os.path.join(self.data_save_path, 'seq_list.pkl'), mode='wb') as file:
			pickle.dump(seq_dict, file)
		with open(os.path.join(self.data_save_path, 'active_meds_list.pkl'), mode='wb') as file:
			pickle.dump(active_profiles_dict, file)
		with open(os.path.join(self.data_save_path, 'active_classes_list.pkl'), mode='wb') as file:
			pickle.dump(active_classes_dict, file)
		with open(os.path.join(self.data_save_path, 'depa_list.pkl'), mode='wb') as file:
			pickle.dump(depa_dict, file)
		with open(os.path.join(self.data_save_path, 'enc_list.pkl'), mode='wb') as file:
			pickle.dump(enc_list, file)
			

###########
# EXECUTE #
###########

if __name__ == '__main__':
	parser = ap.ArgumentParser(description='Preprocess the data extracted from the pharmacy database before input into the machine learning model', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--numyears', metavar='Type_String', type=str, nargs="?", default='5', help='Number of years in the data to process. Defaults to 5')
	parser.add_argument('--sourcefile', metavar='Type_String', type=str, nargs="?", default='paper/data/20050101-20180101.csv', help='Data file load. Defaults to "paper/data/20050101-20180101.csv".')
	parser.add_argument('--definitionsfile', metavar='Type_String', type=str, nargs="?", default='paper/data/definitions.csv', help='Definitions file to provide mappings from medication ids to pharmacological classes. Defaults to "paper/data/definitions.csv".')

	args = parser.parse_args()
	num_years = args.numyears
	source_file = args.sourcefile
	definitions_file = args.definitionsfile

	if not int(num_years):
		logging.critical('Argument --numyears {} is not an integer. Quitting...'.format(num_years))
		quit()
	try:
		if(not os.path.isfile(source_file)):
			logging.critical('Data file: {} not found. Quitting...'.format(source_file))
			quit()
	except TypeError:
		logging.critical('Invalid data file given. Quitting...')
		quit()
	try:
		if(not os.path.isfile(definitions_file)):
			logging.critical('Definitions file: {} not found. Quitting...'.format(definitions_file))
			quit()
	except TypeError:
		logging.critical('Invalid definitions file given. Quitting...')
		quit()

	pp = preprocessor(source_file, definitions_file, restrict_data=num_years)
	pp.preprocess()
