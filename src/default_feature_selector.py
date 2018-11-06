from feature_selector import FeatureSelector

class DefaultFeatureSelector(FeatureSelector):

	def select(self):
		pass

	def training_data(self):	
		return self.data.for_train().gene_data()

	def test_data(self):	
		return self.data.for_test().gene_data()

	def desc(self):
		return 'Default Feature Selector'