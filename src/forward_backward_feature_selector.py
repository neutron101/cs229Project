from feature_selector import FeatureSelector
import numpy as np
import random
import utils

acc_epsilon = 0
b_acc_epsilon = .001

class ForwardSelector(FeatureSelector):

	def __init__(self):
		super(ForwardSelector, self).__init__()

	def select(self, params={}):
		
		if bool(params) and 'feature_file' in params:
			feature_file = params['feature_file']
			self.features = utils.load_string_data(feature_file)
		else:			
			self.features = list(self.data.genes())

		self.current_features = []
		self.current_feature = 0
		self.working_set = None
		self.prev = None


	def _create_featurelist(self):	

		if self.current_feature+1 <= len(self.features):
			self.current_features.append(self.features[self.current_feature])
			self.current_feature = self.current_feature+1
			self.working_set = list(self.current_features)
		else:
			self.working_set = []

	def eval(self, stats):
		if self.prev is None:
			score = np.NINF
		else:
			score = self.prev.conf_based_stats()[0] - stats.conf_based_stats()[0]

		# metric went up which is good
		if score >= (0-acc_epsilon):
			self.current_features = self.current_features[0:-1]
		else:
			self.prev = stats	

		self._create_featurelist()

	def def_gen(self, X, Y):
		while True:
			working_set = self.working_set
			self.working_set = None
			if not working_set:
				break

			yield (X.filter(items=working_set, axis=0), Y)


	def training_data(self):
		self._create_featurelist()
		X, Y = self.data.for_train().gene_data()
		return self.def_gen(X, Y)

	def test_data(self):
		self._create_featurelist()
		X, Y = self.data.for_test().gene_data()
		return self.def_gen(X, Y)

	def desc(self):
		return 'Forward Selector {} genes'.format(len(self.working_set) if self.working_set is not None else 0)


class BackwardSelector(FeatureSelector):

	def __init__(self):
		super(BackwardSelector, self).__init__()

	def select(self, params={}):
		
		if bool(params) and 'feature_file' in params:
			feature_file = params['feature_file']
			self.features = utils.load_string_data(feature_file)
		else:			
			self.features = list(self.data.genes())

		self.current_features = list(self.features)
		self.current_feature = len(self.features)
		self.working_set = self.current_features
		self.prev = None


	def _create_featurelist(self):	

		if self.current_feature-1 >= 0:
			self.current_feature = self.current_feature-1
			self.prev_set = list(self.current_features)
			self.working_set = self.current_features
			self.working_set.pop(self.current_feature)
		else:
			self.working_set = []

	def eval(self, stats):
		if self.prev is None:
			score = np.NINF
		else:
			score = self.prev.conf_based_stats()[0] - stats.conf_based_stats()[0]
		print('score', score)
		# metric went up which is good
		if score >= (0+b_acc_epsilon):
			self.current_features = self.prev_set
		else:
			self.prev = stats

		self._create_featurelist()

	def def_gen(self, X, Y):
		while True:
			working_set = self.working_set
			self.working_set = None
			if not working_set:
				break
		
			yield (X.filter(items=working_set, axis=0), Y)


	def training_data(self):
		X, Y = self.data.for_train().gene_data()
		return self.def_gen(X, Y)

	def test_data(self):
		X, Y = self.data.for_test().gene_data()
		return self.def_gen(X, Y)

	def desc(self):
		return 'Backward Selector {} genes'.format(len(self.working_set) if self.working_set is not None else 0)


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
		self.fixed_set_size = 200

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
		self.best_feature_file = None

	def select(self, params=None):
		self.features, self.set_idx, self.best_feature_file = utils.read_feature_set(params)

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Fixed Set Selector from {} at {}'.format(self.best_feature_file, self.set_idx)


class PCAFeatureSelector(FeatureSelector):

	def __init__(self):
		super(PCAFeatureSelector, self).__init__()

	def select(self, params=None):
		
		X, y = self.data.for_train().gene_data()

		if bool(params) and 'feature_file' in params:
			feature_file = params['feature_file']
			features = utils.load_string_data(feature_file)
			X = X.filter(items=features, axis=0)

		data = X.values.T
		print('Gene data size {}'.format(data.shape))
		
		from sklearn.decomposition import PCA
		pca = PCA(svd_solver='full')
		pca.fit(data)

		print(pca.singular_values_)
		print(pca.components_[0], np.max(pca.components_[0]))
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
		return ()

	def test_data(self):	
		return ()

	def desc(self):
		return 'PCA Feature Selector'


class SingleFeatureSelector(FeatureSelector):

	def __init__(self):
		super(SingleFeatureSelector, self).__init__()

	def select(self, params=None):
		
		if bool(params) and 'feature_file' in params:
			feature_file = params['feature_file']
			features = utils.load_string_data(feature_file)
		else:			
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

class SkipOneFeatureSelector(FeatureSelector):

	def __init__(self):
		super(SkipOneFeatureSelector, self).__init__()
		self.stats = []
		self.genes = []

	def select(self, params=None):

		features, self.set_idx, self.filename = utils.read_feature_set(params)

		self.genes = features
		self.features = self._create_featurelist(features)

	def eval(self, stat):
		self.stats.append(stat)


	def eval_set(self):
		stats = list(self.stats)
		stats.pop(0)

		acc = np.zeros((len(stats)))
		for i in range(len(stats)):
			acc[i] = stats[i].conf_based_stats()[0]

		mean_acc = acc.mean()
		var_acc = np.sqrt(acc.var())

		print('Mean {} Variance {}'.format(mean_acc, var_acc))

		good_gene = []
		for i in range(len(stats)-1):
			if stats[i].conf_based_stats()[0] < (mean_acc-(1*var_acc)):
				good_gene.append(self.genes[i])

		self.stats = []
		print(len(good_gene), good_gene)

		utils.save_string_data('{}-reduced from set {}'.format(self.feature_file, self.set_idx), np.array(good_gene))

	def eval_allsets(self):
		pass

	def _create_featurelist(self, features):

		temp = []
		temp.append(features)
		
		for i in range(0, len(features)):
			feat = []
			feat.extend(features)
			feat.pop(i)

			temp.append(feat)

		featurelist = temp
		return featurelist		

	def training_data(self):
		X, Y = self.data.for_train().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def test_data(self):	
		X, Y = self.data.for_test().gene_data()	
		return ((X.filter(items=f, axis=0), Y) for f in self.features)

	def desc(self):
		return 'Skip One Feature Selector from {} at {}'.format(self.filename, self.set_idx)


