import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

data_dir = 'data/'


def process_data():
    gene_file = 'gene_expression.csv'
    clinical_file = 'clinical.csv'

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
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": range(10, 81, 10)
    }

    clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    clf.score(x_train, y_train)

    print('Accuracy: ' + str (clf.sbest_score_))
    print('Best parameters: ' + str(clf.best_params_))
    return clf.best_params_


def rf_hyperparameter_search(x_train, y_train):
    parameters = {
        'bootstrap': [True, False],
        'max_depth': [30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3, 6],
        'n_estimators': [10, 500, 1000, 1500, 2000]
    }

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    clf.score(x_train, y_train)

    print('Accuracy: ' + str(clf.best_score_))
    print('Best parameters: ' + str(clf.best_params_))
    return clf.best_params_


def linear_hyperparameter_Search(x_train, y_train):
    parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    clf = GridSearchCV(LogisticRegression(), parameters, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    clf.score(x_train, y_train)

    print('Accuracy: ' + str(clf.best_score_))
    print('Best parameters: ' + str(clf.best_params_))
    return clf.best_params_


def main():
    # split into train and test data
    x_train, y_train, x_test, y_test = process_data()
    union_path = data_dir + 'unionall'
    atleast2_path = data_dir + 'atleast2'
    union_all = set(np.loadtxt(union_path, dtype= str))
    intersection_2 = set(np.loadtxt(atleast2_path, dtype= str))
    x_train_union = x_train.loc[:, union_all]
    x_train_intersection = x_train.loc[:, intersection_2]

    # GBM Model selection
    # print('GBM Union Set')
    # gbm_hyperparameter_search(x_train_union, y_train)
    # print('GBM Intersection Set')
    # gbm_hyperparameter_search(x_train_intersection, y_train)
    print(cross_val_score(GradientBoostingClassifier(loss='deviance', subsample=0.6, learning_rate=0.15,
                                                     min_samples_leaf=3, n_estimators=50, min_samples_split=3,
                                                     max_features='log2', max_depth=3), x_train_union, y_train, cv=10).mean())

    print(cross_val_score(GradientBoostingClassifier(loss='deviance', subsample=0.7, learning_rate=0.05,
                                                     min_samples_leaf=1, n_estimators=60, min_samples_split=3,
                                                     max_features='sqrt', max_depth=5), x_train_intersection, y_train, cv=10).mean())

    # rf Model selection
    print('RF Union Set')
    rf_hyperparameter_search(x_train_union, y_train)
    print('RF Intersection Set')
    rf_hyperparameter_search(x_train_intersection, y_train)

    # GBM Model selection
    print('Linear Union Set')
    linear_hyperparameter_Search(x_train_union, y_train)
    print('Linear Intersection Set')
    linear_hyperparameter_Search(x_train_intersection, y_train)


if __name__ == "__main__":
    main()
