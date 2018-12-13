import os
import consts as cs
import pkgutil
import sys
from pydoc import locate
import inspect
from sklearn.metrics import confusion_matrix
from feature_selector import FeatureSelector
from classifier import BaseClassifier
import numpy  as np
from sklearn.feature_selection import VarianceThreshold
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def myprint(str, filename=None):
	if filename is not None:
		with open(os.path.join(cs.output_dir, replace_with_(filename)), 'a') as f:
			f.write(str)
	else:
		print(str)

def load_all_modules_from_dir(dirname, exclusions=[]):
	fs_class_list = []
	cl_class_list = []
	for importer, package_name, _ in pkgutil.iter_modules([dirname]):
		full_package_name = '%s.%s' % (dirname, package_name)
		if full_package_name not in sys.modules:
			module = importer.find_module(package_name).load_module(full_package_name)
			clsmembers = inspect.getmembers(module, inspect.isclass)
			for cl in clsmembers:
				if cl[0] not in exclusions and cl[1].__module__ == full_package_name:
					if issubclass(cl[1], FeatureSelector):
						fs_class_list.append(cl[1])
					if issubclass(cl[1], BaseClassifier):
						cl_class_list.append(cl[1])


	return fs_class_list, cl_class_list

def load_string_data(filename):
	return np.loadtxt(filename, dtype='S', delimiter='||')

def save_string_data(filename, matx):
	np.savetxt(filename, matx, fmt="%s", delimiter='||')

def dictprint(dictvalues):
	return ", ".join("{}={}".format(k, v) for k, v in dictvalues.items()) if dictvalues is not None else ""

def replace_with_(str):
	return str.replace(' ', '_').replace('/', '_')

def read_feature_set(params):

	if bool(params) and 'feature_file' in params:
		feature_file = params['feature_file']
		value = load_string_data(feature_file)

		set_idx = -1
		if not isinstance(value[0], np.ndarray):
			features = []
			features.append(value)
		else:
			if 'index' in params:
				set_idx = params['index']
	
			if set_idx != -1:
				features = [value[set_idx]]

		return features, set_idx, feature_file

	else:			
		raise Exception('Feature file name not specified: ' + dictprint(params))


def remove_near_zero_variance(x_train, x_test, filename=None, thresholdv=.01):
    selector = VarianceThreshold(threshold=thresholdv)
    model = selector.fit(x_train)
    print(x_train.shape)
    nzv_x = x_train[x_train.columns[model.get_support(indices=True)]]
    nzv_x_test = x_test[x_test.columns[model.get_support(indices=True)]]

    feature_space = x_train.columns[model.get_support(indices=True)]
    if filename is not None:
	    np.savetxt(filename, feature_space, fmt='%s')

    print 'Feature count {}'.format(len(feature_space))
    
    return nzv_x, nzv_x_test

def do_error(ds, patients, true, predictions, filename=None):

	if len(true)<= 0 or len(predictions)<=0:
		return

	data = ds.cdf
	chars = ['sex', 'age', 'cohort', 'type_cancer_3', 'bmi', 'weight', 'activity', 'dioxin_like', 'cadmium', 'lead', 'timetodiagnosis', 'sm_status_2']
	data = data.filter(items=chars, axis=1)
	data = data.loc[lambda df: df.get('cohort') == 2, :]

	tp = dict((v, []) for v in data.keys().values)
	tn = dict((v, []) for v in data.keys().values)
	fn = dict((v, []) for v in data.keys().values)
	fp = dict((v, []) for v in data.keys().values)

	for i in range(len(true)):
		t = true[i] == 1
		p = predictions[i] == 1

		if t and p:
			copy(tp, data.T.get(patients[i]))
		elif not t and not p:
			copy(tn, data.T.get(patients[i]))
		elif t and not p:
			copy(fn, data.T.get(patients[i]))
		else:
			copy(fp, data.T.get(patients[i]))

	if filename is not None:
		filename = filename+'_analysis.csv'
		keys, tpstr = merge_with_counts(tp)
		myprint('\t{}\n'.format('\t'.join(keys)), filename)
		myprint('True Postives\t{}\n'.format(tpstr), filename)
		myprint('True Negatives\t{}\n'.format(merge_with_counts(tn)[1]), filename)
		myprint('False Postives\t{}\n'.format(merge_with_counts(fp)[1]), filename)
		myprint('False Negatives\t{}\n'.format(merge_with_counts(fn)[1]), filename)

	ranges = np.array([1, 4, 9, 16, 25, 36])
	plot_range = np.sqrt(ranges[np.where(ranges>=len(chars))[0][0]])
	

	for c_i, kch in enumerate(chars):
		
		n_groups = 4

		bin_names = set().union(tp[kch], tn[kch], fp[kch], fn[kch])

		vals1, counts1 = counting(tp[kch], bin_names)
		vals2, counts2 = counting(tn[kch], bin_names)
		vals3, counts3 = counting(fp[kch], bin_names)
		vals4, counts4 = counting(fn[kch], bin_names)

		v = set()
		v = v.union(vals1)
		v = v.union(vals2)
		v = v.union(vals3)
		v = v.union(vals4)

		index = np.arange(n_groups)
		opacity = 0.8
		bar_width = .5/len(v)

		val_dict = {}
		for i in v:
			
			if i not in val_dict:
				val_dict[i] = []

			if i in vals1:
				val_dict[i].append(counts1[list(vals1).index(i)])
			else:
				val_dict[i].append(0)

			if i in vals2:
				val_dict[i].append(counts2[list(vals2).index(i)])
			else:
				val_dict[i].append(0)

			if i in vals3:
				val_dict[i].append(counts3[list(vals3).index(i)])
			else:
				val_dict[i].append(0)

			if i in vals4:
				val_dict[i].append(counts4[list(vals4).index(i)])
			else:
				val_dict[i].append(0)


		plt.subplot(3,4,c_i+1)
		i = 0
		for k, v in val_dict.items():
			rects1 = plt.bar(index + i*bar_width, v, bar_width,
                 alpha=opacity,
                 label=k)
			i += 1

		
		plt.xlabel(kch)
		plt.ylabel('Counts')
		plt.title('Counts by type')
		plt.xticks(index + bar_width, ('TP', 'TN', 'FP', 'FN'))
		plt.legend()
		 
	plt.show()


	return tp, fp, tn, fn


def copy(d, series):

	if series is not None:
		for k in d.keys():
			value = series.get(k)		
			if isinstance(value, str):
				value = value.lower().strip()
			elif isinstance(value, int):
				value = value
			elif isinstance(value, float):
				if math.isnan(value):
					value = int(-0)
				else:
					value = int(('{:5.0f}'.format(value)))
				
			d[k].append(value)


def merge_with_counts(d):

	keys = d.keys()

	for k in keys:
		v = d[k]
		c = len(v)*1.0
		vals, counts = np.unique(v, return_counts=True)
		counts = counts/c 

		s = ', '.join(['{}:{:2.2f}'.format(vals[i], counts[i]) for i in range(len(vals))])
		d[k] = s

	result = '\t'.join([d[i] for i in keys])

	return keys, result

def counting(lt, all_vs):
	
	vals, counts = np.unique(lt, return_counts=True)
	
	all_vs = [i for i in all_vs if isinstance(i, int)]
	lt = [i for i in lt if isinstance(i, int)]

	if len(all_vs) > 5 and len(lt)>0:		
		counts, vals = np.histogram(lt, bins=np.histogram(all_vs, bins=5)[1])
		nv = []
		for v in range(len(vals)-1):
			nv.append('{}-{}'.format(vals[v], vals[v+1]))
		vals = nv
		
	return vals, counts


