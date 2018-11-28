class FeatureSelector(object):

	def __init__(self):
		None

	def attach_dataset(self, dataset):
		self.data = dataset

	def select(self, params=None):
		 raise NotImplementedError

	def eval(self, stat):
		pass

	def eval_set(self):
		pass

	def eval_allsets(self):
		pass

	def training_data(self):	
		raise NotImplementedError

	def test_data(self):	
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError