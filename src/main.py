import numpy as np
import sys
import argparse
from dataset import Dataset
from utils import myprint, load_all_modules_from_dir
from argparse import RawTextHelpFormatter
from stats import Stats
from mlfactory import MLFactory as mlf

data_dir = '../data/' 
output_dir = '../output'

def main():
	
	###################
	#Installs feature classifier classes stored in the feature and classifier directory 
	install_classes()

	###################
	feature_sel, classifier, output_filename = setup_args()
	# exit()
	###################
	global data_dir
	#create and setup data
	data_dir = args.data_dir if data_dir is None else data_dir
	dataset = Dataset(data_dir)
	dataset.load_gene_data()
	###################

	if feature_sel == 0 and classifier == 0:
		stats = runall(dataset, write_filename=output_filename)
	else:
		#setup feaure selector and classifier
		f_sel, c_sel = map_impl(feature_sel, classifier)
		stats = [run(f_sel, c_sel, dataset, write_filename=output_filename)]
	###################

	print('Finito....')


def run(feature_selector, classifier, dataset, write_filename=None):

	print('Running classifer "{}" with feature selector "{}"'.format(classifier.desc(), feature_selector.desc()))

	feature_selector.attach_dataset(dataset)
	feature_selector.select()

	classifier.with_feature(feature_selector)

	test_data_gen = feature_selector.test_data()

	all_stats = Stats()
	for data in feature_selector.training_data():
		
		Stats.run_timed(lambda :classifier.fit(data), filename=write_filename)
		
		test_data = test_data_gen.next()
		stats = classifier.predict(test_data[0])
		stats.set_printheader('No of. features: {}.\n'.format(data[0].shape[0]))
		stats.record_confusion_matrix(test_data[1])
		stats.itworked(filename=write_filename)

		all_stats.add(stats)

	return all_stats

def runall(dataset, write_filename=None):

	print('Running all classifers with all feature selectors')
	st = []

	for fs in mlf.feature_selectors.keys():
		for cl in mlf.classifiers.keys():
			f_sel, c_sel = map_impl(fs, cl)
			stats = run(f_sel, c_sel, dataset, write_filename)
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
		value = '{}\n{} - > {}'.format(value, key, mlf.feature_selectors[key]().__class__.__name__)
	parser.add_argument('feature', metavar='<feature index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(1,no_of_fselectors+1)],
                    help=value)

	value = ''
	for key in mlf.classifiers.keys():
		value = '{}\n{} - > {}'.format(value, key, mlf.classifiers[key]().__class__.__name__)
	parser.add_argument('classifier', metavar='<classifier index>', type=int, nargs=1, default=0, 
					choices=[idx for idx in range(1,no_of_fselectors+1)],
                    help=value)

	parser.add_argument('-of', metavar='<filename>', nargs=1, default=None, 
                    help='Filename for printing output results')
	args = parser.parse_args()

	feature_sel = args.feature[0]
	classifier = args.classifier[0]
	output_filename = args.of

	return feature_sel, classifier, output_filename[0] if output_filename is not None else output_filename

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