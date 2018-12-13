from classifier import BaseClassifier
from stats import Stats
from sklearn.svm import SVC 
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import utils
import ast

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
			self.params['C'] = float(d('C'))
			self.params['gamma'] = float(d('gamma'))
			self.params['degree'] = float(d('degree'))
			self.params['coef0'] = float(d('coef0'))


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
		self.params = {'kernel': 'rbf', 'C': 1, 'gamma': 1.25, 'degree' : 0, 'coef0' : 0.0}
		
	def fit(self, data, params=None):

		X, y = data
		X = X.values.transpose()

		val = lambda d, key, default: d[key] if key in d else default
		d = lambda l: val(params, l, self.params[l])
		if bool(params):
			self.params['kernel'] = d('kernel')
			self.params['C'] = float(d('C'))
			self.params['gamma'] = float(d('gamma'))
			self.params['degree'] = float(d('degree'))
			self.params['coef0'] = float(d('coef0'))


		self.classifier = SVC(kernel=self.params['kernel'],\
		 C=self.params['C'], \
		  gamma=self.params['gamma'], \
		  degree=self.params['degree'], \
		  coef0=self.params['coef0'], \
		  verbose=False)

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
		return 'SVM {}'.format(utils.dictprint(self.params))


class SVMModelParameterEstimator(BaseClassifier):

	def __init__(self):
		super(SVMModelParameterEstimator, self).__init__()
		self.best_params = {}
		
	def fit(self, data, params=None):

		X, y = data
		X = X.values.transpose()

		val = lambda d, key, default: d[key] if key in d else default
		d = lambda l: val(params, l, None)
		if bool(params):
			tuned_parameters = [
		            {'kernel': [d('kernel')], 'C': [float(d('C'))], 'degree':[float(d('degree'))], 'gamma': [float(d('gamma'))], 'coef0' : [float(d('coef0'))]}]

		else:
			tuned_parameters = [
					{'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1.25, 1, 1.5, 2, 2.5, 3, 3.5, 4], 'C': [.005, .001, .01, .1, .5, 1, 3, 5, 7, 9, 11]},
				#	{'kernel': ['linear'], 'C': [.01, .1, .5, 1, 5, 10, 15]},
		        # {'kernel': ['poly'], 'C': [.005, .001, .01, .1, .5, 1, 3, 5, 7, 10], 'degree':[1,2,3,4], 'gamma': [0.1, 0.5, 1.25, 1, 1.5, 2], 'coef0' : [-1.5, -1, 0, .25, .5, .75, 1,3,5,7]}
		        ]


		self.cv = GridSearchCV(SVC(), tuned_parameters, cv=10,
		               scoring='accuracy', refit = True)

		self.cv.fit(X, y)
		self.classifier = self.cv.best_estimator_
		print(self.classifier, 'Best params', self.cv.best_params_)
		print('Best score: {:3.4f}'.format(self.cv.best_score_))
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
		return 'SVM Model Parameter Estimator {}'.format(utils.dictprint(self.best_params))

