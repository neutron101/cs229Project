import numpy as np
import sys
import argparse
from dataset import Dataset
from utils import myprint, load_all_modules_from_dir
from argparse import RawTextHelpFormatter
from stats import Stats
from mlfactory import MLFactory as mlf
import consts as cs

def main():
	
	#add a new feature reduction class
 	mlf.install_feature_selectors('forward_backward_feature_selector.ForwardBackwardSelector')
 	
 	###################
 	#add classifier to classifer choices
 	mlf.install_classifier('svm.SVM')

	###################
	#Installs feature classifier classes stored in the feature and classifier directory 
	# install_classes()

	###################
	modeling, feature_sel, classifier, output_filename = setup_args()
	# exit()
	###################
	global data_dir
	#create and setup data
	dataset = Dataset(cs.data_dir)
	dataset.load_gene_data()
	###################

	if feature_sel == 0 and classifier == 0:
		stats = runall(dataset, write_filename=output_filename)
	else:
		#setup feaure selector and classifier
		f_sel, c_sel = map_impl(feature_sel, classifier)
		if modeling:
			stats = model_selection(f_sel, c_sel, dataset, write_filename=output_filename)
		else:
			stats = test(f_sel, c_sel, dataset, write_filename=output_filename)
	###################

	print('Finito....')

def model_selection(feature_selector, classifier, dataset, write_filename=None):

	print('Model selection with classifer "{}" with feature selector "{}"'.format(classifier.desc(), feature_selector.desc()))

	feature_selector.attach_dataset(dataset)
	feature_selector.select()

	index = 1
	classifier_stats = Stats()
	for data in feature_selector.training_data():
		
		stats = Stats.run_timed(lambda :classifier.fit(data), filename=write_filename)
		
		stats.set_printheader('No of. features: {}.\n'.format(data[0].shape[0]))
		stats.mystats(filename=write_filename)
		classifier_stats.add_classifier_stat(stats)
		index = index + 1

		if index > 100:
			break

	if write_filename is not None:
		classifier_stats.classifier_stats(write_filename+'_'+classifier.desc()+'_'+feature_selector.desc())

	return classifier_stats

def test(feature_selector, classifier, dataset, write_filename=None):

	print('Testing with classifer "{}" with feature selector "{}"'.format(classifier.desc(), feature_selector.desc()))

	feature_selector.attach_dataset(dataset)
	feature_selector.select()

	test_data_gen = feature_selector.test_data()

	classifier_stats = Stats()
	for data in feature_selector.training_data():
		
		Stats.run_timed(lambda :classifier.fit(data), filename=write_filename)
		
		test_data = test_data_gen.next()
		stats = classifier.predict(test_data[0])
		stats.set_printheader('No of. features: {}.\n'.format(data[0].shape[0]))
		stats.record_confusion_matrix(test_data[1])
		stats.mystats(filename=write_filename)

		classifier_stats.add_classifier_stat(stats)

	return classifier_stats

def runall(dataset, write_filename=None):

	print('Running all classifers with all feature selectors')
	st = []

	for fs in mlf.feature_selectors.keys():
		for cl in mlf.classifiers.keys():
			f_sel, c_sel = map_impl(fs, cl)
			stats = test(f_sel, c_sel, dataset, write_filename)
			st.append(stats)

	return st


def map_impl(f_idx, c_idx):
	f = mlf.create_feature_selector(f_idx)
	c = mlf.create_classifier(c_idx)

	return f, c

def setup_args():

	no_of_fselectors = len(mlf.feature_selectors.keys())
	no_of_cselectors = len(mlf.classifiers.keys())

	parser = argparse.ArgumentParser(description = 'Select feature selector and classifier', \
									formatter_class=RawTextHelpFormatter)
	value = ''
	for key in mlf.feature_selectors.keys():
		value = '{}\n{} - > {}'.format(value, key, mlf.feature_selectors[key])
	parser.add_argument('feature', metavar='<feature index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(1,no_of_fselectors+1)],
                    help=value)

	value = ''
	for key in mlf.classifiers.keys():
		value = '{}\n{} - > {}'.format(value, key, mlf.classifiers[key])
	parser.add_argument('classifier', metavar='<classifier index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(1,no_of_cselectors+1)],
                    help=value)

	parser.add_argument('-of', metavar='<filename>', nargs=1, default=None, 
                    help='Filename for printing output results')

	parser.add_argument('-m',  default=False, 
                    help='Enable model selection', action="store_true")


	args = parser.parse_args()

	feature_sel = args.feature[0]
	classifier = args.classifier[0]
	output_filename = args.of
	modeling = args.m

	return modeling, feature_sel, classifier, output_filename[0] if output_filename is not None else output_filename

def install_classes():

	f_classes = load_all_modules_from_dir('feature', exclusions=['FeatureSelector'])
	c_classes = load_all_modules_from_dir('classifiers', exclusions=['BaseClassifier'])

	assert(len(f_classes) > 0)
	assert(len(c_classes) > 0)

	for f in f_classes:
		mlf.install_feature_selectors(f)
		
	for c in c_classes:
		mlf.install_classifier(c)	


if __name__ == "__main__":
    main()