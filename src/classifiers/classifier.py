class BaseClassifier(object):

	def __init__(self):
		None

	def with_feature(self, feature_sel):
		self.feature_sel = feature_sel

	def fit(self):
		raise NotImplementedError 

	def predict(self):
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError