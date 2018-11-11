import numpy as np
import math
import sys, os, time
from datetime import datetime
import consts as cs
from sklearn.metrics import confusion_matrix
from utils import myprint

np.set_printoptions(precision=4)

class Stats(object):

	def __init__(self):
		self.agg = []

	def add(self, another):
		self.agg.append(another)

	def set_printheader(self, header):
		self.header = header

	def itworked(self, filename=None):
		myprint('-------------Statistics-------------------------------------\n', filename)
		myprint(self.header, filename)
		if len(self.agg) > 0:
			for s in self.agg:
				myprint('Confusion matrix: tn={d[0]:02.4f}, fp={d[1]:02.4f}, fn={d[2]:02.4f}, tp={d[3]:02.4f}\n'.format(d=self.s.cnf_matrix_norm.ravel()),filename)				
		else:
			myprint('Confusion matrix: tn={d[0]:02.4f}, fp={d[1]:02.4f}, fn={d[2]:02.4f}, tp={d[3]:02.4f}\n'.format(d=self.cnf_matrix_norm.ravel()),filename)

		myprint('------------------------------------------------------------\n', filename)


	def set_predictions(self,predictions):
		self.predictions = predictions

	@classmethod
	def run_timed(cl,op,filename=None,verbose=False):
		start = datetime.now()
		stats = Stats.run(op)
		duration = datetime.now() - start

		myprint('Time lapsed: {} secs.\n'.format(duration),filename)

		return stats

	def record_confusion_matrix(self, true):
		assert (self.predictions is not None), "Predictions not set"

		cnf_matrix = confusion_matrix(true, self.predictions)
		self.cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

	@classmethod
	def run(cl,op,write_filename=None,verbose=False):
		return op()

