import numpy as np
import sys
import argparse
from dataset import Dataset

from stats import Stats
from mlfactory import MLFactory as mlf

data_dir = '../data/' 
output_dir = '../output'

def main():
	
	###################
	#add a new feature reduction class
	mlf.install_feature_selectors('default_feature_selector.DefaultFeatureSelector')
	
	###################
	#add classifier to classifer choices
	mlf.install_classifier('naive_bayes_classifier.NaiveBayesClassifier')

	###################
	feature_sel, classifier = setup_args()

	###################
	global data_dir
	#create and setup data
	data_dir = args.data_dir if data_dir is None else data_dir
	dataset = Dataset(data_dir)
	###################

	if feature_sel == 0 and classifier == 0:
		stats = runall(dataset)
	else:
		#setup feaure selector and classifier
		f_sel, c_sel = map_impl(feature_sel, classifier)
		stats = run(f_sel, c_sel, dataset)
	###################

	print(stats.itworked())

	print('Finito')

def setup_args():

	no_of_fselectors = len(mlf.feature_selectors.keys())
	no_of_cselectors = len(mlf.classifiers.keys())

	parser = argparse.ArgumentParser(description = 'Select feature selector and classifier')
	value = ''
	for key in mlf.feature_selectors.keys():
		value = '{}\n{} - > {}'.format(value, key, mlf.feature_selectors[key])
	parser.add_argument('feature', metavar='<feature index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(0,no_of_fselectors)],
                    help=value)

	value = ''
	for key in mlf.classifiers.keys():
		value = '{}\n{} - > {}'.format(value, key, mlf.classifiers[key])
	parser.add_argument('classifier', metavar='<classifier index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(0,no_of_fselectors)],
                    help=value)

	args = parser.parse_args()

	feature_sel = args.feature[0]
	classifier = args.classifier[0]

	return feature_sel, classifier

def run(feature_selector, classifier, dataset):

	print('Running classifer "{}"" with feature selector "{}"'.format(feature_selector.desc(), classifier.desc()))

	feature_selector.attach_dataset(dataset)
	feature_selector.select()

	classifier.with_feature(feature_selector)
	classifier.fit()
	stats = classifier.predict()

	return stats

def runall(dataset):

	print('Running all classifers with all feature selectors')
	st = Stats()

	for fs in mlf.feature_selectors.keys():
		for cl in mlf.classifiers.keys():
			f_sel, c_sel = map_impl(fs, cl)
			stats = run(f_sel, c_sel, dataset)
			st.add(stats)

	return st


def map_impl(f_idx, c_idx):
	f = mlf.create_feature_selector(f_idx)
	c = mlf.create_classifier(c_idx)

	return f, c

if __name__ == "__main__":
    main()