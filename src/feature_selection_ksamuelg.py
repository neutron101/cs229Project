import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def process_data():
    gene_file = 'gene_expression.csv'
    clinical_file = 'clinical.csv'
    data_dir = '../data/'

    gene_file_path = os.path.join(data_dir, gene_file)
    clinical_file_path = os.path.join(data_dir, clinical_file)

    gene_df = pd.read_csv(gene_file_path, header=0, index_col=0)
    gene_df = gene_df.transpose()
    clinical_df = pd.read_csv(clinical_file_path, header=0, index_col=0)

    clinical_df = clinical_df.loc[:, ['cohort', 'type_cancer_3']]

    df = gene_df.join(clinical_df)

    df['type_cancer_3'] = (df['type_cancer_3'] == 'CO').astype(int)

    X_df = df.loc[:, 'cg00009553': 'cohort']
    cohort = df.loc[:, 'cohort']
    Y_df = df.loc[:, 'type_cancer_3']

    X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=77, stratify=cohort)
    x_train_cohort = X_train.loc[:, 'cohort']
    X_train = X_train.loc[:, 'cg00009553': 'cg27666046']
    X_test = X_test.loc[:, 'cg00009553': 'cg27666046']

    return X_train, y_train, X_test, y_test


def lasso_feature_selection(x_train, y_train, x_test, y_test):
    clf = linear_model.Lasso(alpha=0.002, fit_intercept=False, selection='cyclic')
    clf.fit(x_train, y_train)
    sfm = SelectFromModel(clf, prefit=True)

    clf_imp = linear_model.Lasso(alpha=0.002, fit_intercept=False, selection='cyclic')
    X_important_train = sfm.transform(x_train)
    X_important_test = sfm.transform(x_test)
    clf_imp.fit(X_important_train, y_train)

    test_pred = clf.predict(x_test)
    test_pred = np.array(test_pred)
    super_threshold_indices = test_pred < 0.5
    test_pred[super_threshold_indices] = 0
    super_threshold_indices = test_pred >= 0.5
    test_pred[super_threshold_indices] = 1

    imp_pred = clf_imp.predict(X_important_test)
    super_threshold_indices = imp_pred < 0.5
    imp_pred[super_threshold_indices] = 0
    super_threshold_indices = imp_pred >= 0.5
    imp_pred[super_threshold_indices] = 1

    acc = accuracy_score(y_test, test_pred)
    imp_only = accuracy_score(y_test, imp_pred)

    print('lasso accuracy: ' + str(acc) + ' lasso importance only: ' + str(imp_only))

    gene_names = x_train.columns[sfm.get_support()]
    return gene_names


def remove_near_zero_variance(x_train, x_test):
    selector = VarianceThreshold(threshold=0.01)
    model = selector.fit(x_train)
    nzv_x = x_train[x_train.columns[model.get_support(indices=True)]]
    nzv_x_test = x_test[x_test.columns[model.get_support(indices=True)]]

    feature_space = x_train.columns[model.get_support(indices=True)]
    output_dir = '../gene_set/'
    filename = output_dir + 'non_zero_variance'
    np.savetxt(filename, feature_space, fmt='%s')
    print len(feature_space)
    return nzv_x, nzv_x_test


def save_best_genes(set_normal, set_nzv):
    output_dir = '../gene_set/'
    for key in set_normal.keys():
        filename = output_dir + key + '_normal'
        np.savetxt(filename, set_normal[key], fmt='%s')

    for key in set_nzv.keys():
        filename = output_dir + key + '_nzv'
        np.savetxt(filename, set_nzv[key], fmt='%s')


def rf_feature_selection(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=1000, min_samples_split=3, min_samples_leaf=3)
    clf.fit(x_train, y_train)
    sfm = SelectFromModel(clf, threshold=0.0025)
    sfm.fit(x_train, y_train)

    clf_imp = RandomForestClassifier(n_estimators=1000, min_samples_split=3, min_samples_leaf=3)
    X_important_train = sfm.transform(x_train)
    X_important_test = sfm.transform(x_test)
    clf_imp.fit(X_important_train, y_train)

    test_pred = clf.predict(x_test)
    imp_pred = clf_imp.predict(X_important_test)

    acc = accuracy_score(y_test, test_pred)
    imp_only = accuracy_score(y_test, imp_pred)

    print('rf accuracy: ' + str(acc) + ' rf importance only: ' + str(imp_only))

    gene_names = x_train.columns[sfm.get_support()]
    return gene_names


def gbm_hyperparameter_search(x_train, y_train):
    # Run for hyperparameter search takes ~30min to converge
    # results
    # {'loss': 'deviance', 'learning_rate': 0.05, 'min_samples_leaf': 3, 'n_estimators': 40, 'min_samples_split': 3,
    #  'max_features': 'sqrt', 'max_depth': 8}
    parameters = {
        "loss": ["deviance"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_samples_split": [3, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_depth": [3, 5, 8],
        "max_features": ["log2", "sqrt"],
        "n_estimators": range(10, 81, 10)
    }

    clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    clf.score(x_train, y_train)
    return(clf.best_params_)


def gbm_feature_selection(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=40, min_samples_leaf=3, max_depth=8)
    clf.fit(x_train, y_train)
    sfm = SelectFromModel(clf, threshold=0.0025)
    sfm.fit(x_train, y_train)

    clf_imp = GradientBoostingClassifier(learning_rate=0.05, n_estimators=40, min_samples_leaf=3, max_depth=8)
    X_important_train = sfm.transform(x_train)
    X_important_test = sfm.transform(x_test)
    clf_imp.fit(X_important_train, y_train)

    test_pred = clf.predict(x_test)
    imp_pred = clf_imp.predict(X_important_test)

    acc = accuracy_score(y_test, test_pred)
    imp_only = accuracy_score(y_test, imp_pred)

    print('gbm accuracy: ' + str(acc) + ' gbm importance only: ' + str(imp_only))

    gene_names = x_train.columns[sfm.get_support()]
    return gene_names


def main():
    # split into train and test data
    x_train, y_train, x_test, y_test = process_data()

    # Perform feature selection without removing near zero variance features first
    print('Running feature selection BEFORE non-zero variance reduction')
    # gene_set_dict = {}
    # gene_set_dict['lasso'] = lasso_feature_selection(x_train, y_train, x_test, y_test)
    # gene_set_dict['random_forest'] = rf_feature_selection(x_train, y_train, x_test, y_test)
    # gene_set_dict['boosting'] = gbm_feature_selection(x_train, y_train, x_test, y_test)

    # Perform feature selection after removing near zero variance features
    print('Running feature selection AFTER non-zero variance reduction')
    nzv_gene_set_dict = {}
    nzv_x_train, nzv_x_test = remove_near_zero_variance(x_train, x_test)
    # nzv_gene_set_dict['lasso'] = lasso_feature_selection(nzv_x_train, y_train, nzv_x_test, y_test)
    # nzv_gene_set_dict['random_forest'] = rf_feature_selection(nzv_x_train, y_train, nzv_x_test, y_test)
    # nzv_gene_set_dict['boosting'] = gbm_feature_selection(nzv_x_train, y_train, nzv_x_test, y_test)

    # Save best gene sets for further processing
    # save_best_genes(gene_set_dict, nzv_gene_set_dict)


if __name__ == "__main__":
    main()

