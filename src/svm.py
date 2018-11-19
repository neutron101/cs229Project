from classifier import BaseClassifier
from stats import Stats
from sklearn.svm import SVC 
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import utils

class SVM(BaseClassifier):

	def __init__(self):
		super(SVM, self).__init__()
		self.params = {'kernel': 'rbf', 'C': 1, 'gamma': 1, 'degree' : 0, 'coef0' : 0.0}
		
	def fit(self, data, params):

		X, y = data
		X = X.values.transpose()

		val = lambda d, key, default: d[key] if key in d else default
		d = lambda l: val(params, l, self.params[l])
		if bool(params):
			self.params['kernel'] = d('kernel')
			self.params['C'] = d('C')
			self.params['gamma'] = d('gamma')
			self.params['degree'] = d('degree')
			self.params['coef0'] = d('coef0')

		self.classifier = SVC(kernel=self.params['kernel'],\
		 C=self.params['C'], \
		  gamma=self.params['gamma'], \
		  degree=self.params['degree'], \
		  coef0=self.params['coef0'], \
		  verbose=False)

		self.classifier.fit(X, y)

		stat = Stats()
		stat.set_confusion_matrix(confusion_matrix(y, self.classifier.predict(X)))
		
		return stat


	def predict(self, t_X):
		stats = Stats()

		t_X = t_X.values.transpose()
		predictions = self.classifier.predict(t_X)

		stats.set_predictions(predictions)

		return stats

	def desc(self):
		return 'SVM {}'.format(utils.dictprint(self.params))


class FeatureSelectorSVM(BaseClassifier):

	def __init__(self):
		super(FeatureSelectorSVM, self).__init__()
		self.kernel = 'rbf'
		self.C = 1
		self.gamma = 1.25
		
	def fit(self, data, params=None):

		X, y = data
		X = X.values.transpose()

		self.classifier = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,verbose=False)

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
		return 'SVM {} kernel C={} gamma={}'.format(self.kernel, self.C, self.gamma)


class SVMModelParameterEstimator(BaseClassifier):

	def __init__(self):
		super(SVMModelParameterEstimator, self).__init__()
		self.best_params = 'TBD'
		
	def fit(self, data, params=None):

		X, y = data
		X = X.values.transpose()

		tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1.25, 1, 1.5, 2],
		             'C': [1, 5, 10, 50, 100]},
		            {'kernel': ['linear'], 'C': [1, 5, 10, 50, 100]},
		            {'kernel': ['poly'], 'C': [1, 5, 10, 50, 100], 'degree':[2,3,4], 'gamma': [0.1, 0.5, 1.25, 1, 1.5, 2], 'coef0' : [0,1,10]}]

		self.cv = GridSearchCV(SVC(), tuned_parameters, cv=5,
		               scoring='accuracy', refit = True)

		self.cv.fit(X, y)
		self.classifier = self.cv.best_estimator_
		print(self.classifier, 'Best params', self.cv.best_params_)
		self.best_params = self.cv.best_params_

		predictions = self.cv.predict(X) 
		conf_matrix = confusion_matrix(y, predictions)
		stat = Stats()
		stat.set_confusion_matrix(conf_matrix)
		
		return stat


	def predict(self, t_X):
		stats = Stats()

		t_X = t_X.values.transpose()
		predictions = self.cv.predict(t_X)

		stats.set_predictions(predictions)

		return stats

	def desc(self):
		return 'SVM Model Parameter Estimator'.format(self.best_params)

