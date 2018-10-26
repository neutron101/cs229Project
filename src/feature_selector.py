
class FeatureSelector(object):

	def __init__(self):
		None

	def attach_dataset(self, dataset):
		self.data = dataset.raw

	def select(self):
		 raise NotImplementedError

	def features(self):	
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError