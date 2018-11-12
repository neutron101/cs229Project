from classifier import BaseClassifier
from stats import Stats
from sklearn.svm import SVC 
from sklearn.model_selection import cross_validate
import numpy as np

class SVM(BaseClassifier):

	def __init__(self):
		super(SVM, self).__init__()
		
	def fit(self, data):

		X, y = data
		X = X.values.transpose()

		self.classifier = SVC(kernel='linear',verbose=False)

		results = cross_validate(self.classifier.fit(X, y), X, y,
                            scoring=self.conf_matrix_score, cv=5)

		result = np.array([[np.sum(results['test_tn']), np.sum(results['test_fp'])], \
			[np.sum(results['test_fn']), np.sum(results['test_tp'])]])

		stat = Stats()
		stat.set_confusion_matrix(result)
		
		return stat


	def predict(self, t_X):
		stats = Stats()

		t_X = t_X.values.transpose()
		predictions = self.classifier.predict(t_X)

		stats.set_predictions(predictions)

		return stats

	def desc(self):
		return 'SVM'


class RBFSVM(BaseClassifier):

	def __init__(self):
		None
		
	def fit(self, data):

		X, y = data
		X = X.values.transpose()

		self.classifier = SVC(kernel='linear',verbose=False)

		self.classifier.fit(X, y)


	def predict(self, t_X):
		stats = Stats()

		t_X = t_X.values.transpose()
		predictions = self.classifier.predict(t_X)

		stats.set_predictions(predictions)

		return stats

	def desc(self):
		return 'SVM'

