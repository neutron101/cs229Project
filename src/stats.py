import numpy as np
import math
import sys, os, time
from datetime import datetime
import consts as cs
from sklearn.metrics import confusion_matrix
from utils import myprint
import plot_utils as pu

np.set_printoptions(precision=4)

class Stats(object):

	def __init__(self):
		self.classifier_stat = []
		self.classifiers_stat = {}
		self.metrics = {}

	def set_printheader(self, header):
		self.header = header

	def mystats(self, filename=None, cond=None):
	
		if cond is None or cond is not None and cond(self):
			myprint('-------------Statistics-------------------------------------\n', filename)
			if '_time' in self.metrics:
				myprint('Time lapsed: {} secs.\n'.format(self.metrics['_time']),filename)
			myprint(self.header, filename)

			myprint('Accuracy={d[0]:3.6f} Sensitivity={d[1]:3.3f} Specificity={d[2]:3.3f} f1_score={d[3]:3.3f} \n'.format(d=self.conf_based_stats()),filename)
			myprint('Confusion matrix: tn={d[0]:02.4f}, fp={d[1]:02.4f}, fn={d[2]:02.4f}, tp={d[3]:02.4f} \n'.format(d=self.cnf_matrix.ravel()),filename)				
			myprint('Normalized Confusion matrix: tn={d[0]:02.4f}, fp={d[1]:02.4f}, fn={d[2]:02.4f}, tp={d[3]:02.4f} \n'.format(d=self.cnf_matrix_norm.ravel()),filename)

			myprint('------------------------------------------------------------\n', filename)

	def classifier_stats(self, filename=None, title=None):
		sensitivity = []
		specificity = []

		if len(self.classifier_stat) > 0:
			metrics = {}
			for st in self.classifier_stat:
				if st.conf_based_stats() is not None:
					sensitivity.append(st.conf_based_stats()[1])
					specificity.append(1-st.conf_based_stats()[2])

				if bool(st.metrics):
					for key, value in st.metrics.items():
						if not key.startswith('_'):
							if key not in metrics:
								metrics[key] = []
							metrics[key].append(value)

			if len(sensitivity) > 0 and len(specificity) > 0:
				pu.plotscatter(specificity, sensitivity, filename, '1-Specificity', 'Sensitivity', title)

			if bool(metrics):
				pu.plotscatter(np.linspace(1,len(metrics.values()[0]), len(metrics.values()[0])), \
					metrics.values(), filename+'_'+key, 'Set', metrics.keys(), 'Metrics')
				

	def classifiers_stats(self, filename=None):
		pass
			

	def set_predictions(self,predictions):
		self.predictions = predictions

	@classmethod
	def run_timed(cl,op,verbose=False):
		start = datetime.now()
		result = Stats.run(op)
		duration = datetime.now() - start

		result.add_metric('{} secs'.format(duration), '_time')
		return result

	def record_confusion_matrix(self, true):
		assert (self.predictions is not None), "Predictions not set"

		self.cnf_matrix = confusion_matrix(true, self.predictions)
		self.cnf_matrix_norm = self.cnf_matrix.astype('float') / self.cnf_matrix.sum(axis=1)[:, np.newaxis]

	def set_confusion_matrix(self, matrix):
		self.cnf_matrix = matrix
		self.cnf_matrix_norm = self.cnf_matrix.astype('float') / self.cnf_matrix.sum(axis=1)[:, np.newaxis]

	@classmethod
	def run(cl,op,write_filename=None,verbose=False):
		return op()

	def conf_based_stats(self):
		c_mat = self.cnf_matrix

		tn, fp, fn, tp = c_mat.ravel() * 1.0
		accuracy = (tp+tn)/(tn+fp+fn+tp)

		sensitivity = tp/(tp+fn)
		specificity = tn/(tn+fp)

		precision = tp/(tp+fp)
		f1_score = (2*precision*sensitivity)/(precision+sensitivity)

		my_score = (tp+tn)/fn if fn>0 else 0

		return accuracy, sensitivity, specificity, f1_score, my_score


	def add_classifier_stat(self, another):
		self.classifier_stat.append(another)

	def add_classifiers_stat(self, another, classifier_name):
		self.classifiers_stat[classifier_name] = another

	def add_metric(self, value, metric):
		if metric not in self.metrics:
			self.metrics[metric] = []

		self.metrics[metric].append(value)
