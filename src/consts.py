gene_file = 'gene_expression.csv'
clinical_file = 'clinical.csv'
output_dir = '../output/'
data_dir = '../data/'

best_features_file = 'BEST'

dataset_filter = {}

gene_cutoff = 3821

#### Modelling parameters

model_selector_params = {
					     'PCAFeatureSelector' : {'feature_file' : 'target_features'}, \
					     'SingleFeatureSelector' : {'feature_file' : 'target_features'}, \

					     'ForwardSelector' : {}, \

					     'BackwardSelector' : {'feature_file' : '../gene_set/unionall'}, \
					     # 'BackwardSelector' : {'feature_file' : 'non_zero_variance'}, \
					     
					     'FixedSetSelector' : {'feature_file' : '../gene_set/atleast2'}, \
					     #'FixedSetSelector' : {'_iter' : ({"feature_file":"target_features_11_18_23", "index":i} for i in range(3))}, \

					     'SkipOneFeatureSelector' : {'_iter' : ({"feature_file":"random_selected_features_1", "index":i} for i in range(23))}}

# model_classifier_params = { 'SVM' : [{'kernel': 'rbf', 'C': .5, 'gamma': .1}] }


# model_classifier_params = { 'FeatureSelectorSVM' : [{'kernel': 'rbf', 'C': 1, 'gamma': 1}, {'kernel': 'rbf', 'C': 1, 'gamma': 1.5}, {'kernel': 'rbf', 'C': 3, 'gamma': .5}] }

# model_classifier_params = { '_exec_mode':'', \
# 		'SVM' : [{'kernel': 'poly', 'C': '1', 'coef0': '0', 'degree': '2', 'gamma': '0.1'}, \
# 		{'kernel': 'poly', 'C': '1', 'coef0': '0', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '0', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '1', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '2'}, {'kernel': 'rbf', 'C': '1', 'gamma': '2'}, {'kernel': 'rbf', 'C': '10', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '10', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '10', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '5', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '5', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '5', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '5', 'gamma': '1'}]}

# model_classifier_params = {'SVM' : [{'kernel': 'poly', 'C': .1, 'coef0':1, 'gamma':1.25, 'degree':2}]}

model_classifier_params = {} #{'SVMModelParameterEstimator' : [{'kernel': 'poly', 'C': 1, 'coef0':7, 'gamma':.1, 'degree':2}]}

################################# Testing parameters #################################

# test_selector_params = { 'FixedSetSelector' : {'_iter' : ({"feature_file":"random_selected_features_1", "index":i} for i in range(35))} }

test_selector_params = { 'FixedSetSelector' : {'feature_file' : '../gene_set/atleast2'} }


test_classifier_params = {'SVM' : [{'kernel': 'poly', 'C': .1, 'coef0':0, 'gamma':1.25, 'degree':2}]}

# test_classifier_params = {'SVM' : [{'kernel': 'rbf', 'C': 5, 'gamma' : 0.1}]}
# test_classifier_params = {'SVM' : [{'kernel': 'linear', 'C': 1}]}

# Models for selecting the best feature set and model combination

# test_classifier_params = {'_exec_mode':'one2one', \
#  'SVM' : [{'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '5', 'gamma': '0.5'}, {'kernel': 'poly', 'C': '1', 'coef0': '1', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'linear', 'C': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '3', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '3', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '2'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.5'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '2'}, {'kernel': 'rbf', 'C': '9', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '9', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '2'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '7', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '3', 'gamma': '0.1'}, {'kernel': 'poly', 'C': '1', 'coef0': '10', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'rbf', 'C': '3', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '5', 'gamma': '1'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'poly', 'C': '1', 'coef0': '1', 'degree': '2', 'gamma': '0.1'}, {'kernel': 'linear', 'C': '5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '1', 'gamma': '1.25'}, {'kernel': 'rbf', 'C': '3', 'gamma': '0.5'}, {'kernel': 'rbf', 'C': '9', 'gamma': '0.1'}] }

