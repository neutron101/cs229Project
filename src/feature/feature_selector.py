class FeatureSelector(object):

	def __init__(self):
		None

	def attach_dataset(self, dataset):
		self.data = dataset

	def select(self, params={}):
		 raise NotImplementedError

	def training_data(self):	
		raise NotImplementedError

	def test_data(self):	
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError