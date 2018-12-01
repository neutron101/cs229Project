import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
gene_file = 'gene_expression.csv'
clinical_file = 'clinical.csv'
output_dir = '../output/'
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

X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=77, stratify = cohort)
x_train_cohort = X_train.loc[:, 'cohort']
X_train = X_train.loc[:, 'cg00009553': 'cg27666046']
X_test = X_test.loc[:, 'cg00009553': 'cg27666046']

# from sklearn import metrics
#
# rcf = RandomForestClassifier()
# rcf.fit(X_train, y_train)
# pred = rcf.predict(X_test)
#
# accuracy = metrics.accuracy_score(y_test, pred)
# print accuracy

from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
# lasso = linear_model.Lasso(alpha=0, fit_intercept = False, selection='cyclic')
# # lasso.fit(X_train, y_train)
# # pred = lasso.predict(X_test)
# # print pred
# #
# # accuracy = metrics.accuracy_score(y_test, pred)
# #
# # print accuracy
from sklearn.ensemble import ExtraTreesClassifier


# forest = ExtraTreesClassifier(n_estimators=250,
#                               random_state=0)
#
# forest.fit(X_train, y_train)
# importances = forest.feature_importances_
#
# feat_importances = pd.Series(forest.feature_importances_, index=X_train.columns)
# feat_importances.nlargest(20).plot(kind='barh')
#
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# # indices = indices[:len(indices/2): 1]
#
# # X_train = X_train[indices]
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices],
#        color="purple", yerr=std[indices], align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlim([-1, X_train.shape[1]])
#
# print range(X_train.shape[1])
# plt.show()

# print(X_train.shape)
#

lasso = linear_model.Lasso(alpha=0.002, fit_intercept = False, selection='cyclic').fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_new = model.transform(X_train)
print(X_new.shape)
# print(X_new)
print(model.get_support())

gene_names = X_train.columns[model.get_support()]

print(gene_names)
np.savetxt('../gene_set/lasso_gene_list', gene_names, fmt='%s')





# cols = list(df.columns)
# for col in cols:
#     X_train[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
#
#
#
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# plt.cla()
# pca = decomposition.PCA(n_components=3)
# pca.fit(X_train)
# X = pca.transform(X_train)
#
# # for name, label in [('Control', 0), ('Lymphoma', 1)]:
# #     ax.text3D(X[y_train == label, 0].mean(),
# #               X[y_train == label, 1].mean() + 1.5,
# #               X[y_train == label, 2].mean(), name,
# #               horizontalalignment='center',
# #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
#
# # Reorder the labels to have colors matching the cluster results
# # y_train = np.choose(y_train, [1, 0]).astype(np.float)
#
# my_cmap = ListedColormap(sns.light_palette("green"), 2)
# plt.scatter(X[x_train_cohort == 1, 0], X[x_train_cohort == 1, 1], c=y_train[x_train_cohort == 1], cmap=my_cmap,
#            edgecolor='k')
#
# plt.colorbar(ticks=range(2), label='Outcome', spacing='proportional')
#
#
# other_cmap = ListedColormap(sns.color_palette("Purples"), 2)
#
#
# plt.scatter(X[x_train_cohort == 2, 0], X[x_train_cohort == 2, 1], c=y_train[x_train_cohort == 2], cmap=other_cmap,
#            edgecolor='k')
#
# plt.colorbar(ticks= [0, 1],cmap = other_cmap, spacing = 'proportional')
#
# plt.title('Principal Component Analysis on Train Data')
# plt.xlabel('1st Principal Component')
# plt.ylabel('2nd Principal Component')
#
# # ax.w_xaxis.set_ticklabels([])
# # ax.w_yaxis.set_ticklabels([])
# # ax.w_zaxis.set_ticklabels([])
#
# plt.show()