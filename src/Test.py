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
import matplotlib.pyplot as plt
import types

np.set_printoptions(threshold=np.nan, linewidth= np.nan)

def main():


	# ds = Dataset('../data/')
	# ds.load_gene_data()

	# print(utils.do_error(ds, ['NS002LY'], [1], [1]))

	# data = 'cg13985765||cg19849557||cg07430472||cg15271348||cg03040554||cg07315018||cg15197657||cg13377965||cg23678210||cg08307171||cg13921622||cg24137774||cg05800339||cg24524379||cg14553824||cg00405789||cg20851559||cg12800028||cg27370471||cg06097704||cg01335597||cg17052552||cg10194536||cg27224718||cg15777028||cg02947253||cg07724896||cg07541443||cg07895657||cg10100266||cg23173806||cg04630334||cg10805039||cg25117345||cg02607489||cg08076462||cg11410080||cg07299736||cg05359518||cg06607825||cg09866409||cg01784351||cg10236687||cg17625535||cg24191264||cg07211044||cg20133191||cg18789685||cg27000503||cg19537719||cg00267196||cg07533417||cg19504605||cg12651540||cg04272736||cg10031651||cg25489617||cg09398138||cg25408707||cg06892439||cg03529555||cg03553781||cg14303079||cg16640553||cg21160290||cg09861075||cg02380064||cg04496895||cg13511777||cg08728613||cg05144540||cg12274479||cg27243624||cg16460157||cg17140034||cg03798162||cg08906366||cg25154609||cg23839398||cg25236028||cg23283423||cg23961973||cg16210979||cg06423354||cg12149606||cg02899473||cg23036385||cg01402746||cg13320898||cg23978473||cg10078009||cg03657281||cg04977856||cg22160185||cg25344401||cg07679219||cg00599124||cg04914283||cg07824824||cg00245465||cg01676996||cg24462992||cg05143420||cg11433787||cg03436208||cg23330281||cg11123583||cg04327675||cg02722214||cg18666249||cg13583272||cg09003527||cg05512099||cg23346128||cg12040834'.split('||')
	# for d in data:
	# 	print(d)



	# compare_genes('.', '../output/BEST_model___SVM_kernel=rbf,_C=7.0,_coef0=0.0,_degree=0.0,_gamma=0.1_Backward_Selector_0_genes')
	# check_cov('../output', 'BEST')

	# combine_data('../output', 'BEST_model', 'all_genes_combined-forwardselected')

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

	# X, Y = ds.for_train().gene_data()
	# value = utils.load_string_data('target_features_11_18_23_combined')
	# X = X.filter(items=value, axis=0)
	# utils.remove_near_zero_variance(X.T, X.T, 'target_features_11_18_23_combined'+'_nzv', thresholdv=.002)

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

	# sel = ForwardSelector()
	# sel.attach_dataset(ds)

	# sel.select() #params = {'st_index': i})


	# for v in sel.training_data():
	# 	i = 0
	# 	for X,Y in v:
	# 		mt = confusion_matrix([1, 0, 1, 0], [1, 1, 0, 0])
	# 		st = Stats()
	# 		st.cnf_matrix = mt
	# 		sel.eval(st)
	# 		i = i+1
		
	# 	print(sel.eval_set())
	# 	print('vecs', i)



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

	# data = utils.load_string_data('target_features_11_18_23_combined')
	# print(type(data))
	# dset = set()
	# for d in data:
	# 	print(type(d))
	# 	dset.add(d)

	# print(data.shape, len(dset))
	# utils.save_string_data('target_features_11_18_23_combined_1', np.array(list(dset)))


def combine_data(dir, prefix, saveto):

	files = os.listdir(dir)
	dset = set()

	for f in files:
		if f.startswith(prefix):
			value = utils.load_string_data(os.path.join(dir,f))
			if isinstance(value, list) or isinstance(value, np.ndarray):
				for v in value:
					if isinstance(v, list) or isinstance(v, np.ndarray):
						dset = dset.union(v)
					else:
						dset.add(v)
			else:
				dset.add(value)

	# print(len(dset), dset)
	utils.save_string_data(saveto, np.array(list(dset)))

def compare_genes(dir, prefix):

	# files = os.listdir(dir)
	sets = set()
	alt = 0
	gene_map = {}
	# for f in files:
		# if f.startswith(prefix):
	value = utils.load_string_data(os.path.join(dir,prefix))
	gene_map[prefix] = set(list(value))

	if len(value.shape) > 0:
		sets = sets.union(value)
		alt = alt + value.shape[0]
	else:
		sets.add(str(value))
		alt = alt + 1

	print(len(sets))

	ds = Dataset('../data/')
	ds.load_gene_data()			
	X, y = ds.for_train().gene_data()
	X = X.filter(items=sets, axis=0)
	data = X.values.T

	from sklearn.decomposition import PCA
	pca = PCA(svd_solver='full')
	pca.fit(data)

	sing_values = pca.singular_values_

	print(sing_values[0:20])

	agg = 0
	for i in range(1, len(sing_values)):
		agg = agg + sing_values[i-1]
		if (agg/np.sum(sing_values)) > .9:
			break
	print('90 percent variance captured by {} vectors'.format(i))

	agg = 0
	for i in range(1, len(sing_values)):
		agg = agg + sing_values[i-1]
		if (agg/np.sum(sing_values)) > .99:
			break
	print('99 percent variance captured by {} vectors'.format(i))

	evec1 = pca.components_[0]

	# disp = np.empty((evec1.shape[0],1), dtype=[('genes', 'U20'),('evec1',np.float64), ('final', np.bool)])
	# index = 0
	# for g in ds.genes():
	# 	disp[index,0] = (g, evec1[index], g in sets)
	# 	index = 1 + index

	# np.set_printoptions(threshold=np.nan, linewidth= np.nan)
	# print(np.sort(disp, axis=0, order='evec1'))

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


def check_cov(dir, prefix):

	files = os.listdir(dir)
	sets = set()
	alt = 0
	for f in files:
		if f.startswith(prefix):
			value = utils.load_string_data(os.path.join(dir,f))

			if len(value.shape) > 0:
				sets = sets.union(value)
				alt = alt + value.shape[0]
			else:
				sets.add(str(value))
				alt = alt + 1
			
			print(len(sets))

	print('Total', alt)

	np.set_printoptions(threshold=np.nan, linewidth= np.nan)
	
	ds = Dataset('../data/')
	ds.load_gene_data()
	X, Y = ds.for_train().gene_data()
	fil = X.filter(items=sets, axis=0)
	print(fil.shape)
	fil_v = fil.values
	cov = np.cov(fil_v)
	print(cov.shape)

	print(np.sort(np.diag(cov)))
	# exit()

	cov[np.where(np.identity(cov.shape[0])==1)] = 0

	
	sort_ind = np.argsort(np.sum(np.abs(cov), axis=1))
	print(sort_ind)
	print(np.sort(np.sum(np.abs(cov), axis=1)))

	l_set = np.array(list(sets))
	trim_set = l_set[sort_ind[0:cov.shape[0]]]
	print(trim_set)

	np_trim_set = np.array(trim_set)
	# utils.save_string_data('test_target_features', np_trim_set)


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