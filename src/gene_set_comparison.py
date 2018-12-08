from matplotlib_venn import venn2, venn3, venn3_circles
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold


boosting_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/boosting_normal', dtype= str))
boosting_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/boosting_nzv', dtype= str))

lasso_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/lasso_normal', dtype= str))
lasso_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/lasso_nzv', dtype= str))

rf_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/random_forest_normal', dtype= str))
rf_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/random_forest_nzv', dtype= str))

svm_list = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/svm_c7_g0.1', dtype= str))
knn_list = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/knn_30final_startingall', dtype= str))
top_3 = ['cg05512099', 'cg23961973', 'cg07596054']

nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/non_zero_variance', dtype= str))

# plt.figure()
# plt.title('Boosting Comparison')
# venn2([boosting_normal, boosting_nzv], set_labels=('Normal', 'NZV'))
#
#
# plt.figure()
# plt.title('Lasso Comparison')
# venn2([lasso_normal, lasso_nzv], set_labels=('Normal', 'NZV'))
#
# plt.figure()
# plt.title('Random Forest Comparison')
# venn2([rf_normal, rf_nzv], set_labels=('Normal', 'NZV'))
#
# plt.figure()
# plt.title('All 3 models')
# venn3([boosting_nzv, lasso_nzv, rf_nzv], set_labels=('Boosting NZV', 'Lasso NZV', 'Random Forest NZV'))
#
# plt.figure()
# plt.title('All 3 Group Members')
# venn3([svm_list, knn_list, boosting_normal], set_labels=('SVM', 'KNN', 'Boosting Normal'))
# gene_file = 'gene_expression.csv'
# clinical_file = 'clinical.csv'
# output_dir = '../output/'
# data_dir = '../data/'
#
# gene_file_path = os.path.join(data_dir, gene_file)
# clinical_file_path = os.path.join(data_dir, clinical_file)
#
# gene_df = pd.read_csv(gene_file_path, header=0, index_col=0)
# gene_df = gene_df.transpose()
# clinical_df = pd.read_csv(clinical_file_path, header=0, index_col=0)
#
# clinical_df = clinical_df.loc[:, ['cohort', 'type_cancer_3']]
#
# df = gene_df.join(clinical_df)
#
# df['type_cancer_3'] = (df['type_cancer_3'] == 'CO').astype(int)
# lut = dict(zip(df['cohort'].unique(), "rbg"))
#
# row_colors = df['cohort'].map(lut)
# can_colors = df['type_cancer_3'].map(lut)
#
# df = df.loc[:, 'cg00009553': 'cg27666046']
# can_colors = df['cohort'].map(lut)
#
#
# # plt.figure()
# # sns.boxplot(x="type_cancer_3", y="cg05512099", hue="type_cancer_3", data=df, palette="Set1")
# # fig, ax = plt.subplots(figsize=(8,6))
# # bp = df.groupby('type_cancer_3').plot(kind='kde', ax=ax)
# lut = dict(zip(df['type_cancer_3'].unique(), "rbg"))
# row_colors = df['type_cancer_3'].map(lut)
# sns.clustermap(df.loc[:, boosting_normal], metric='euclidean', row_colors = row_colors)

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

# X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=77, stratify=cohort)
# x_train_cohort = X_train.loc[:, 'cohort']

x_cohort_1 = df[df['cohort'] == 1]
x_cohort_2 = df[df['cohort'] == 1]
x_cohort_1 = x_cohort_1.loc[:, 'cg00009553': 'cg27666046']
x_cohort_2 = x_cohort_2.loc[:, 'cg00009553': 'cg27666046']

# venn2([rf_normal, rf_nzv], set_labels=('Normal', 'NZV'))

selector = VarianceThreshold(threshold=0.01)
model = selector.fit(x_cohort_1)
cohort_1_feature_space = x_cohort_1.columns[model.get_support(indices=True)]
print ('cohort 1 feature space:' + str(len(cohort_1_feature_space)))
output_dir = '../gene_set/'
filename = output_dir + 'non_zero_variance_cohort_1'
np.savetxt(filename, cohort_1_feature_space, fmt='%s')
cohort_1_feature_space = set(np.loadtxt('../gene_set/non_zero_variance_cohort_1', dtype= str))


selector = VarianceThreshold(threshold=0.01)
model = selector.fit(x_cohort_2)
cohort_2_feature_space = x_cohort_2.columns[model.get_support(indices=True)]
print ('cohort 2 feature space:' + str(len(cohort_2_feature_space)))
output_dir = '../gene_set/'
filename = output_dir + 'non_zero_variance_cohort_2'
np.savetxt(filename, cohort_2_feature_space, fmt='%s')
cohort_2_feature_space = set(np.loadtxt('../gene_set/non_zero_variance_cohort_2', dtype= str))

plt.figure()
venn2([cohort_1_feature_space, cohort_2_feature_space], set_labels=('cohort 1', 'cohort 2'))

plt.figure()
venn3([cohort_1_feature_space, cohort_2_feature_space, nzv], set_labels=('cohort 1', 'cohort 2', 'All NZv'))
plt.show()
