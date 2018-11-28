import numpy as np
from dataset import Dataset
from forward_backward_feature_selector import ForwardSelector, BackwardSelector
import os
# from svm import SVM as svm
import utils
# from mlxtend.feature_selection import SequentialFeatureSelector as ss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from stats import Stats
from sklearn.metrics import confusion_matrix

def main():

	# ds = Dataset('../data/')
	# ds.load_gene_data()
	# X, Y = ds.for_train().gene_data()
	# sets = np.loadtxt('target_features', dtype='S', delimiter='||')
	# X = X.filter(items=sets, axis=0)

	# forward(X.T,Y)
	# utils.load_all_modules_from_dir('feature',exclusions=['FeatureSelector'])
	# values = utils.load_all_modules_from_dir('.',exclusions=[]) #['BaseClassifier', 'FeatureSelector'])
	# print(locals())

	# ds = Dataset('../data/')
	# ds.load_gene_data()

	# clean_genes(ds)


	# value = utils.load_string_data('random_selected_features_1-reduc')
	# print(value, len(value))
	# X, Y = ds.for_train().gene_data()
	# # for f in X.axes[0]:
	# # 	print ('--{}--'.format(f))
	# # exit()	
	# # print(value[1][197])
	# for f in value:
	# 	fil = X.filter(items=f, axis=0)
 # 		if fil.shape[0] != 200:
 # 			print(f, fil.axes[0], fil.shape[0])
 # 			# for a in fil.axes[0]:
 # 			# 	print(a)
 # 			exit()

	# print(ds.genes())
	# print(ds.for_train().gene_data()[0].shape, ds.for_train().gene_data()[1].shape)

	# sel = BackwardSelector()
	# sel.attach_dataset(ds)
	# sel.select()
	# for X,Y in sel.training_data():
	# 	print(X.shape)
	# 	mt = confusion_matrix([1, 0, 1, 0], [1, 1, 0, 0])
	# 	st = Stats()
	# 	st.cnf_matrix = mt
		# sel.eval(st)



	# clf = svm()
	# clf.with_feature(sel)
	# clf.fit()

	# items = {'a':'d', 'e':'f'}
	# print(", ".join("{}={}".format(k, v) for k, v in items.items()))


	# value = np.loadtxt('random_selected_features_1', dtype='S', delimiter='||')
	# np.savetxt('TEST DATA 1', value, fmt="%s", delimiter='||')
	# print(value.shape)
	# alls = set()
	# np.set_printoptions(threshold=np.nan, linewidth= np.nan)
	# counts = np.zeros((value.shape[0], value.shape[0]))
	# for i in range(value.shape[0]):
	# 	s1 = set(value[i])
	# 	for j in range(i, value.shape[0]): 
	# 		s2 = set(value[j])
	# 		counts[i][j] = len(s1.intersection(s2))

	# 	# alls = alls.union(s)

	# # print('All', len(alls))
	# print(counts)

	data = utils.load_string_data('target_features_11_18_23')
	dset = set()
	for d in data:
		dset = dset.union(set(d))

	print(data.shape, len(dset))


def extract_params():

	import re
	with open('../output/parse', 'r') as f:
		lines = f.readlines()
		par = []
		for l in lines:
			d = {}
			for p in [l for l in  l.split(',')]:
				k,v = p.split('=')
				d[k.strip()] = v.strip().rstrip('.').rstrip('\n')

			# print(d)
			par.append(d)
		
		print(par)



def test_gen():
	global a
	while a is not None:
		yield a
		

def check_cov():

	files = os.listdir('.')
	sets = set()
	alt = 0
	for f in files:
		if f.startswith('random_selected_features_1-reduced'):
			value = utils.load_string_data(f)

			if len(value.shape) > 0:
				sets = sets.union(value)
				alt = alt + value.shape[0]
			else:
				sets.add(str(value))
				alt = alt + 1
			
			print(len(sets))

	print('Total', alt)

	ds = Dataset('../data/')
	ds.load_gene_data()
	X, Y = ds.for_train().gene_data()
	fil = X.filter(items=sets, axis=0)
	print(fil.shape)
	fil_v = fil.values
	cov = np.cov(fil_v)
	print(cov.shape)

	# print(np.sort(np.diag(cov)))
	# exit()

	cov[np.where(np.identity(cov.shape[0])==1)] = 0

	np.set_printoptions(threshold=np.nan, linewidth= np.nan)
	sort_ind = np.argsort(np.sum(np.abs(cov), axis=1))
	print(sort_ind)
	print(np.sort(np.sum(np.abs(cov), axis=1)))

	l_set = np.array(list(sets))
	trim_set = l_set[sort_ind[0:410]]
	print(len(trim_set))

	np_trim_set = np.array(trim_set)
	utils.save_string_data('test_target_features', np_trim_set)


def forward(X, Y):

	from mlxtend.feature_selection import SequentialFeatureSelector as SFS
	knn = KNeighborsClassifier(n_neighbors = 35)
	sv = SVC(kernel='poly', C=1, degree=2, coef0=10, gamma=0.1, verbose=False)

	cls = sv
	sfs = SFS(cls, 
			k_features=20,
			forward=True, 
			floating=False, 
			scoring='accuracy',
			cv=10,
			verbose=2
			)
	sfs = sfs.fit(X, Y)

	print('\nSequential Forward Selection:')
	print(sfs.k_feature_idx_)
	print('CV Score:')
	print(sfs.k_score_)

def clean_genes(ds):

	genes = set(ds.genes())
	print(len(genes))
	data = utils.load_string_data('random_selected_features_0')
	new_data = []
	for d in data:
		good = []
		for v in d:
			if v.startswith('cg'):
				good.append(v)

		small = list(genes.difference(good))
		wanted = d.shape[0] - len(good)

		want = list(np.random.choice(len(small), wanted))

		for w in want:
			good.append(small[w])

		new_data.append(good)

	utils.save_string_data('random_selected_features_0-clean', np.array(new_data))



#####

# grep Selector ../output/model__* -h | cut -f 2,9,10,11,12,13 -d ' ' | awk '{print substr($0,5)}' | sort -n | cut -d ' ' -f2- > ../output/parse

####

if __name__ == "__main__":
    main()