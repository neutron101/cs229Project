import numpy as np
from dataset import Dataset
from forward_backward_feature_selector import ForwardBackwardSelector
import os

def main():

	ds = Dataset('../data/')
	data = ds.load_gene_data()

	# print(ds.genes())
	# print(ds.for_train().gene_data()[0].shape, ds.for_train().gene_data()[1].shape)

	sel = ForwardBackwardSelector()
	sel.attach_dataset(ds)
	sel.select({'f' : ''})
	test = sel.training_data()


if __name__ == "__main__":
    main()