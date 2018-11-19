gene_file = 'gene_expression.csv'
clinical_file = 'clinical.csv'
output_dir = '../output/'
data_dir = '../data/'

best_features_file = 'BEST'

dataset_filter = {}

param_forwardbackwardselector = {'forward' : False}

#### Modelling parameters

model_selector_params = {'SingleFeatureSelector' : [{'None': 'None'}]}

model_classifier_params = {'SVM' : [{'kernel': 'rbf', 'C': 1, 'gamma': 1.25}]}


#### Testing parameters

test_selector_params = {'FixedSetSelector' : [{'best_feature_file': 'target_features'}]}

test_classifier_params = {}
# Models for selecting the best feature set and model combination
#test_classifier_params = {'SVM' : [{'kernel': 'rbf', 'C': 10, 'gamma': 2}, {'kernel': 'rbf', 'C': 1, 'gamma': 2}, {'kernel': 'rbf', 'C': 1, 'gamma': 1.5}, {'kernel': 'rbf', 'C': 1, 'gamma': 1}, {'kernel': 'rbf', 'C': 5, 'gamma': 0.5}, {'kernel': 'rbf', 'C': 1, 'gamma': 2}, {'kernel': 'rbf', 'C': 1, 'gamma': 1.5}, {'kernel': 'rbf', 'C': 5, 'gamma': 0.5}, {'kernel': 'rbf', 'C': 1, 'gamma': 1.5}]}

