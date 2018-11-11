import os
import consts as cs
import pkgutil
import sys
from pydoc import locate
import inspect

def myprint(str, filename=None):
	if filename is not None:
		with open(os.path.join(cs.output_dir, filename), 'a') as f:
			f.write(str)
	else:
		print(str)

def load_all_modules_from_dir(dirname,exclusions=[]):
	class_list = []
	for importer, package_name, _ in pkgutil.iter_modules([dirname]):
		full_package_name = '%s.%s' % (dirname, package_name)
		if full_package_name not in sys.modules:
			module = importer.find_module(package_name).load_module(full_package_name)
			clsmembers = inspect.getmembers(module, inspect.isclass)
			for cl in clsmembers:
				if cl[0] not in exclusions and cl[1].__module__ == full_package_name:
					class_list.append(cl[1])

	return class_list