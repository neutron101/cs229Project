from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

class BaseClassifier(object):

	def __init__(self):
		self.conf_matrix_score = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn), \
				'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}

	# Input: (X, y)
	#
	# returns predictions from k-fold cross validation 
	# [[-1 1 ... -1 1] ... [-1 1 ... 1 1]]
	def fit(self, data, params=None):
		raise NotImplementedError 

	# Input: (X, y)
	#
	# returns predictions
	# [-1 1 ... -1 1]
	def predict(self, data):
		raise NotImplementedError

	def desc(self):
		raise NotImplementedError

