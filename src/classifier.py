from sklearn.metrics import make_scorer
import utils as ut

class BaseClassifier(object):

	def __init__(self):
		self.conf_matrix_score = {'tp' : make_scorer(ut.tp), 'tn' : make_scorer(ut.tn), \
				'fp' : make_scorer(ut.fp), 'fn' : make_scorer(ut.fn)}

	# Input: (X, y)
	#
	# returns predictions from k-fold cross validation 
	# [[-1 1 ... -1 1] ... [-1 1 ... 1 1]]
	def fit(self, data):
		raise NotImplementedError 

	# Input: (X, y)
	#
	# returns predictions
	# [-1 1 ... -1 1]
	def predict(self, data):
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError

