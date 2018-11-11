from classifier import BaseClassifier
from stats import Stats
from sklearn.svm import SVC 

class SVM(BaseClassifier):

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


