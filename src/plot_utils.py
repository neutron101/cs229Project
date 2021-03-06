from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import os
import consts as cs
from scipy.interpolate import spline
import numpy as np


def plot_confusion_matrix(self,cm, classes=['Cancer','Healthy'],
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plotline(x, y, save_filename, xlabel, ylabel, title=None):
 
    markers = ['.', ',', 'o', '^', '*', '+', '-', '>', '<', 'x']
    markerfacecolors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    if not isinstance(y, list):
        print('Expecting a list of y values')
        return

    if len(y) > 0 and not isinstance(y[0], list):
        y = [y]


    # Plot dataset
    plt.figure()
    index = 0
    for ys in y:
        plt.plot(x, ys, marker=markers[2], color=markerfacecolors[index], \
            markersize=6, markerfacecolor=markerfacecolors[np.random.choice(10,1)[0]], \
            label=ylabel, linestyle='-')
        index = index + 1
    
    plt.title(title)
    plt.legend()

    # Add labels and save to disk
    plt.xlabel(xlabel)
    plt.savefig(os.path.join(cs.output_dir, save_filename)+'.png')


def plotscatter(x, y, save_filename, xlabel, ylabel, title=None):
 
    markers = ['.', ',', 'o', '^', '*', '+', '-', '>', '<', 'x']
    markerfacecolors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    if not isinstance(y, list):
        print('Expecting a list of y values')
        return

    if len(y) > 0 and not isinstance(y[0], list):
        y = [y]


    # Plot dataset
    plt.figure()
    index = 0
    for ys in y:
        plt.scatter(x, ys, marker=markers[index], c=markerfacecolors[index], \
            s=6, \
            label=ylabel[index])
        index = index + 1
    
    plt.title(title)
    plt.legend()

    # Add labels and save to disk
    plt.xlabel(xlabel)
    plt.savefig(os.path.join(cs.output_dir, save_filename)+'.png')

