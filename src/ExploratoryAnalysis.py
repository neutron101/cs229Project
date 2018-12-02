import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

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

pca = decomposition.PCA(n_components=3)
pca.fit(X_train)
X = pca.transform(X_train)

# 3d Plot
# for name, label in [('Control', 0), ('Lymphoma', 1)]:
#     ax.text3D(X[y_train == label, 0].mean(),
#               X[y_train == label, 1].mean() + 1.5,
#               X[y_train == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
# y_train = np.choose(y_train, [1, 0]).astype(np.float)

my_cmap = ListedColormap(sns.light_palette("green"), 2)
plt.scatter(X[x_train_cohort == 1, 0], X[x_train_cohort == 1, 1], c=y_train[x_train_cohort == 1], cmap=my_cmap,
           edgecolor='k')

plt.colorbar(ticks=range(2), label='Outcome', spacing='proportional')


other_cmap = ListedColormap(sns.color_palette("Purples"), 2)


plt.scatter(X[x_train_cohort == 2, 0], X[x_train_cohort == 2, 1], c=y_train[x_train_cohort == 2], cmap=other_cmap,
           edgecolor='k')

plt.colorbar(ticks= [0, 1],cmap = other_cmap, spacing = 'proportional')

plt.title('Principal Component Analysis on Train Data')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

plt.show()