import numpy as np
from dataset import Dataset
# from forward_backward_feature_selector import ForwardBackwardSelector
import os
# from svm import SVM as svm
import utils

def main():

	f = "{'2' : [1, 2], 'b' : [3, 3]}"
	print('_dd'.startswith('_'))


	print(dict(f))

	# utils.load_all_modules_from_dir('feature',exclusions=['FeatureSelector'])
	# values = utils.load_all_modules_from_dir('.',exclusions=[]) #['BaseClassifier', 'FeatureSelector'])
	# print(locals())

	# ds = Dataset('../data/')
	# ds.load_gene_data()


	# value = utils.load_string_data('random_selected_features_1')
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

	# sel = ForwardBackwardSelector()
	# sel.attach_dataset(ds)
	# sel.select({'f' : ''})
	# test = sel.training_data()

	# clf = svm()
	# clf.with_feature(sel)
	# clf.fit()

	# items = {'a':'d', 'e':'f'}
	# print(", ".join("{}={}".format(k, v) for k, v in items.items()))


	# value = np.loadtxt('target_features', dtype='S', delimiter='||')
	# # np.savetxt('TEST DATA 1', value, fmt="%s", delimiter='||')
	# print(value.shape)
	# for i in value:
	# 	print(i)

	# print(utils.replace_with_('i d dd'))

if __name__ == "__main__":
    main()