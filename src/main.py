import numpy as np
import sys
import argparse
from dataset import Dataset
from utils import myprint, load_all_modules_from_dir
from argparse import RawTextHelpFormatter
from stats import Stats
from mlfactory import MLFactory as mlf
import consts as cs
from feature_selector import FeatureSelector
from classifier import BaseClassifier
import utils
import os

config = dict(globals(  ))
execfile('consts.py', config)

def main():

	from pydoc import locate
	#add a new feature reduction class
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.ForwardBackwardSelector'))
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.RandomLimitedFeatureSelector'))
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.ForwardFixedSetSelector'))
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.FixedSetSelector'))
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.PCAFeatureSelector'))
 	mlf.install_feature_selectors(locate('forward_backward_feature_selector.SingleFeatureSelector'))
 	
 	# ###################
 	#add classifier to classifer choices
 	mlf.install_classifier(locate('svm.SVM'))
 	mlf.install_classifier(locate('svm.FeatureSelectorSVM'))
 	mlf.install_classifier(locate('svm.SVMModelParameterEstimator'))

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
			output_filename = 'model_{}'.format(output_filename)
			stats = model_selection(f_sel, c_sel, dataset, write_filename=output_filename)
		else:
			output_filename = 'test_{}'.format(output_filename)
			stats = test(f_sel, c_sel, dataset, write_filename=output_filename)
	###################

	print('Finito....')

def model_selection(feature_selector, classifier, dataset, write_filename=None):

	feature_selector.attach_dataset(dataset)

	if write_filename is not None:
		write_filename = write_filename+'_'+classifier.desc()+'_'+feature_selector.desc()

	feature_sel_name = feature_selector.__class__.__name__
	classifier_name = classifier.__class__.__name__

	sel_params_list = config.get('model_selector_params').get(feature_sel_name) if config.get('model_selector_params').get(feature_sel_name) is not None else [{}]
	cl_params_list = config.get('model_classifier_params').get(classifier_name) if config.get('model_classifier_params').get(classifier_name) is not None else [{}]

	all_classifier_stats = Stats()
	for cl_param in cl_params_list:

		classifier_stats = Stats()

		best=[]
		for feature_param in sel_params_list:	
			
			feature_selector.select(feature_param)

			f_idx = 0
			for data in feature_selector.training_data():
				f_idx = f_idx+1
				stats = Stats.run_timed(lambda :classifier.fit(data, cl_param), filename=write_filename)
				
				print('Finished model selection with classifer "{}" with feature selector "{}"'.format(classifier.desc(), feature_selector.desc()))

				stats.set_printheader(stat_header({'FeatureCount':data[0].shape[0], 'Set':f_idx}, classifier.desc()))
				stats.mystats(filename=write_filename)
				classifier_stats.add_classifier_stat(stats)

				## Thresholding for good feature set selection which will be saved later
				if stats.conf_based_stats()[1] > .70 and (1-stats.conf_based_stats()[2]) < .35:
					X, _ = data
					fea = X.axes[0].values
					best.append(fea)

			utils.save_string_data(os.path.join(config.get('output_dir'), config.get('best_features_file')), best)

			if write_filename is not None:
				classifier_stats.classifier_stats(filename=utils.replace_with_(write_filename), title='{} \n {}'.format(classifier.desc(),feature_selector.desc()))

		all_classifier_stats.add_classifiers_stat(classifier_stats, classifier.desc())

	return all_classifier_stats


def test(feature_selector, classifier, dataset, write_filename=None):

	feature_selector.attach_dataset(dataset)

	feature_sel_name = feature_selector.__class__.__name__
	classifier_name = classifier.__class__.__name__

	sel_params_list = config.get('test_selector_params').get(feature_sel_name) if config.get('test_selector_params').get(feature_sel_name) is not None else [{}]
	cl_params_list = config.get('test_classifier_params').get(classifier_name) if config.get('test_classifier_params').get(classifier_name) is not None else [{}]

	all_classifier_stats = Stats()
	for cl_param in cl_params_list:

		classifier_stats = Stats()

		for feature_param in sel_params_list:	
			
			feature_selector.select(feature_param)

			test_data_gen = feature_selector.test_data()		

			f_idx = 0
			for data in feature_selector.training_data():
				f_idx = f_idx+1 

				Stats.run_timed(lambda :classifier.fit(data, cl_param), filename=write_filename)
				
				print('Finished testing with classifer "{}" with feature selector "{}"'.format(classifier.desc(), feature_selector.desc()))

				test_data = test_data_gen.next()
				stats = classifier.predict(test_data[0])
				stats.set_printheader(stat_header({'FeatureCount':data[0].shape[0], 'Set':f_idx}, classifier.desc()))
				stats.record_confusion_matrix(test_data[1])

				#writes data to a file/console if the optional condition is true
				stats.mystats(filename=write_filename, cond=lambda s: s.conf_based_stats()[0]>.72)

				classifier_stats.add_classifier_stat(stats)

		all_classifier_stats.add_classifiers_stat(classifier_stats)

	return all_classifier_stats


def runall(dataset, write_filename=None):

	print('Running all classifers with all feature selectors')
	st = []

	for fs in mlf.feature_selectors.keys():
		for cl in mlf.classifiers.keys():
			f_sel, c_sel = map_impl(fs, cl)
			stats = test(f_sel, c_sel, dataset, write_filename)
			st.append(stats)

	return st


########################################################

#					UTILS

########################################################

def stat_header(featurep, classifier_desc):
	sel_params = utils.dictprint(featurep)
	return 'Selector: {} with {}.\n'.format(sel_params, classifier_desc)


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

	f_classes, c_classes = load_all_modules_from_dir('.', exclusions=['FeatureSelector', 'BaseClassifier'])

	assert(len(f_classes) > 0)
	assert(len(c_classes) > 0)

	f_classes.sort()
	c_classes.sort()

	for f in f_classes:
		mlf.install_feature_selectors(f)
		
	for c in c_classes:
		mlf.install_classifier(c)


if __name__ == "__main__":
    main()