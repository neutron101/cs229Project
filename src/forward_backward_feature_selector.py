from feature_selector import FeatureSelector
import numpy as np
import random
import utils

class ForwardBackwardSelector(FeatureSelector):

	def __init__(self):
		super(ForwardBackwardSelector, self).__init__()

	def select(self, params={'forward':False}):
		
		features = list(self.data.genes())

		if not params['forward']:
			features.reverse()

		self.features = self._create_featurelist(features)

		print('No of {} feature options {}. Examples: {}'.format('forward' if params['forward'] else 'backward', \
			len(self.features), self.features[0:4]))

	def _create_featurelist(self, features):

		temp = [[]]
		
		for i in range(1, len(features)+1):
			element = features[i-1]
			new_list = list(temp[i-1])
			new_list.append(element)
			temp.append(new_list)

		featurelist = temp[1:]
		return featurelist		

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Forward and Backward Selector'


class RandomLimitedFeatureSelector(FeatureSelector):

	def __init__(self):
		super(RandomLimitedFeatureSelector, self).__init__()
		self.num_features = 200
		self.samples = 5000

	def select(self, params=None):
		
		features = list(self.data.genes())

		self.features = self._create_featurelist(features)


	def _create_featurelist(self, features):

		temp = []
		
		feat = np.array(features)

		for i in range(1, self.samples):
			element = np.random.choice(len(features), self.num_features, replace=False)
			element = feat[element]
			
			temp.append(element)

		featurelist = temp
		return featurelist		

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Random {} Selector from {} samples'.format(self.num_features, self.samples)


class ForwardFixedSetSelector(FeatureSelector):

	def __init__(self):
		super(ForwardFixedSetSelector, self).__init__()
		self.fixed_set_size = 100

	def select(self, params=None):
		
		features = list(self.data.genes())

		self.features = self._create_featurelist(features)


	def _create_featurelist(self, features):

		temp = []
		
		feat = np.array(features)
		for i in range(1, len(features)-self.fixed_set_size+1):
			element = np.copy(feat[i-1:self.fixed_set_size+i-1])
			temp.append(element)

		featurelist = temp
		return featurelist		

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Forward Fixed Set={} Selector'.format(self.fixed_set_size)

class FixedSetSelector(FeatureSelector):

	def __init__(self):
		super(FixedSetSelector, self).__init__()
		self.best_feature_file = 'BEST'

	def select(self, params):
		self.best_feature_file = params['best_feature_file'] if params is not None and 'best_feature_file' in params else self.best_feature_file	
		value = utils.load_string_data(self.best_feature_file )
		self.features = list(value)

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Fixed Set Selector from {}'.format(self.best_feature_file)

class PCAFeatureSelector(FeatureSelector):

	def __init__(self):
		super(PCAFeatureSelector, self).__init__()

	def select(self, params=None):
		
		X, y = self.data.for_train().gene_data()
		data = X.values 
		print('Gene data size {}'.format(data.shape))
		
		from sklearn.decomposition import PCA
		pca = PCA(svd_solver='full')
		pca.fit(data)

		print(pca.singular_values_)
		sing_values = pca.singular_values_
		agg = 0
		for i in range(1, len(sing_values)):
			agg = agg + sing_values[i-1]
			if (agg/np.sum(sing_values)) > .9:
				break
		print('90 percent variance captured by {} vectors'.format(i))

		agg = 0
		for i in range(1, len(sing_values)):
			agg = agg + sing_values[i-1]
			if (agg/np.sum(sing_values)) > .99:
				break
		print('99 percent variance captured by {} vectors'.format(i))

	def training_data(self):
		return ([], [])

	def test_data(self):	
		return ([],[])

	def desc(self):
		return 'PCA Feature Selector'


class SingleFeatureSelector(FeatureSelector):

	def __init__(self):
		super(SingleFeatureSelector, self).__init__()

	def select(self, params=None):
		
		features = list(self.data.genes())

		self.features = self._create_featurelist(features)

	def _create_featurelist(self, features):

		temp = []
		
		for i in range(1, len(features)+1):
			new_list = [features[i-1]]
			temp.append(new_list)

		featurelist = temp
		return featurelist		

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Single Feature Selector'


