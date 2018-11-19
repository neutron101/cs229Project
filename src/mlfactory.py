import importlib

class MLFactory(object):

	classifiers = {}
	feature_selectors = {}

	sel_idx = 0
	sel_cl_idx = 0

	@classmethod
	def install_feature_selectors(cl, clazz):
		MLFactory.sel_idx = MLFactory.sel_idx + 1
		MLFactory.feature_selectors[MLFactory.sel_idx] = clazz
		return MLFactory.sel_idx

	@classmethod
	def install_classifier(cl, clazz):
		MLFactory.sel_cl_idx = MLFactory.sel_cl_idx + 1
		MLFactory.classifiers[MLFactory.sel_cl_idx] = clazz
		return MLFactory.sel_cl_idx

	@classmethod
	def create_feature_selector(cl, idx):
		from pydoc import locate
		my_class = cl.feature_selectors[idx]
		return my_class()

	@classmethod
	def create_classifier(cl, idx):
		from pydoc import locate
		my_class = cl.classifiers[idx]
		return my_class()
