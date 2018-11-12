from feature_selector import FeatureSelector

class ForwardBackwardSelector(FeatureSelector):

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
		return 'Forward and Backward Feature Selector'