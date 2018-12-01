import os
import numpy as np
import pandas as pd
import consts as cs
from sklearn.model_selection import train_test_split
from utils import myprint

config = dict(globals(  ))
execfile('consts.py', config)

class Dataset(object):


	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.train = True

	def load_gene_data(self):
		gene_file_path = os.path.join(self.data_dir, cs.gene_file)
		
		df = pd.read_csv(gene_file_path, header=0, index_col=0)

		cutoff = config.get('gene_cutoff')
		self.gene_ids = list(df.axes[0].values[0:cutoff])
		self.patient_ids = list(df.axes[1].values)
		self.data = df.values

		print('Loaded data for {} genes. Examples: {}....'.format(len(self.gene_ids), self.gene_ids[0:4]))
		print('Loaded data for {} patients. Examples: {}....'.format(len(self.patient_ids), self.patient_ids[0:4]))

		clinical_file_path = os.path.join(self.data_dir, cs.clinical_file)
		cdf = pd.read_csv(clinical_file_path, header=0, index_col=0)

		print('Loaded clinial data with {} features. Examples: {}....'.format(cdf.shape[1], cdf.axes[1].values[0:4]))

		agg = cdf.join(df.T)

		for col, value in cs.dataset_filter.items():
			myprint('Filtering dataset on column \"{}\"" with value {}'.format(col,value))
			agg = agg.loc[lambda df: df.get(col) == value, :]
		
		self.train_set, self.test_set = train_test_split(agg, test_size=0.2, \
										random_state=77, \
										stratify=agg.get('cohort').values)


	def genes(self):
		return self.gene_ids

	def gene_data(self):
		data = self.get_data()
		X, Y = data.get(self.gene_ids).T, self.convert_labels(data)

		return X, Y

	def patients(self):
		return self.patient_ids

	def for_train(self):
		self.train = True
		return self

	def for_test(self):
		self.train = False
		return self

	#################################
	def convert_labels(self, agg):
		return np.where(agg.get('type_cancer_3').values == 'CO', -1, 1)

	def get_data(self):
		return self.train_set if self.train else self.test_set