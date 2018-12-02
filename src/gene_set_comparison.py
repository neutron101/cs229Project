from matplotlib_venn import venn2, venn3, venn3_circles
from matplotlib import pyplot as plt
import numpy as np

boosting_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/boosting_normal', dtype= str))
boosting_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/boosting_nzv', dtype= str))

lasso_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/lasso_normal', dtype= str))
lasso_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/lasso_nzv', dtype= str))

rf_normal = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/random_forest_normal', dtype= str))
rf_nzv = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/random_forest_nzv', dtype= str))

svm_list = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/svm_c7_g0.1', dtype= str))
knn_list = set(np.loadtxt('/Users/Karina/Developer/cs229Project/gene_set/final_features_knn.txt', dtype= str))


plt.figure()
plt.title('Boosting Comparison')
venn2([boosting_normal, boosting_nzv], set_labels=('Normal', 'NZV'))


plt.figure()
plt.title('Lasso Comparison')
venn2([lasso_normal, lasso_nzv], set_labels=('Normal', 'NZV'))

plt.figure()
plt.title('Random Forest Comparison')
venn2([rf_normal, rf_nzv], set_labels=('Normal', 'NZV'))

plt.figure()
plt.title('All 3 models')
venn3([boosting_normal, lasso_nzv, rf_nzv], set_labels=('Boosting Normal', 'Lasso NZV', 'Random Forest NZV'))

plt.figure()
plt.title('All 3 Group Members')
venn3([svm_list, knn_list, boosting_normal], set_labels=('SVM', 'KNN', 'Boosting Normal'))

plt.show()
