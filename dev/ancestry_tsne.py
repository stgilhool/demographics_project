from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cag_input_data
import numpy as np
import pandas as pd
import csv
import pdb
from mpl_toolkits.mplot3d import Axes3D

def labels_to_colors(labels_arr):
    ''' Convert n_labels x 5 one-hot vectors to 1-D list of n_labels colors '''
    color_vec1 = ['black', 'blue', 'purple', 'cyan']
    color_vec2 = ['gray', 'red', 'orange', 'yellow']
    output = []
    for row in labels_arr:
        if row[-1] == 1:
            color_i = [c for (c,lab) in zip(color_vec2,row[:-1]) if lab]
        else:
            color_i = [c for (c,lab) in zip(color_vec1,row[:-1]) if lab]
        output = np.append(output, color_i)

    return output

set_3d = False

# Loading dataset
data, labels = cag_input_data.preprocess_data(input_type='pca', n_data=10)

# Transform labels to colors for plotting
clrs = labels_to_colors(labels)

# Plot PCA data
#plt.subplot(1,2,1)
x_pca = data[:,0]
y_pca = data[:,1]
z_pca = data[:,2]

fig = plt.figure()
plotsize = 10
if set_3d:
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x_pca, y_pca, z_pca, c=clrs, s=plotsize)
else:
    ax = fig.add_subplot(121)
    ax.scatter(x_pca, y_pca, c=clrs, s=plotsize)
#plt.scatter(x_pca, y_pca, z_pca, c=clrs)
ax.set_title("PCA")
# Defining Model
if set_3d:
    model = TSNE(3, learning_rate=100)
else:
    model = TSNE(learning_rate=100, perplexity=25)
# Fitting Model
transformed = model.fit_transform(data)

# Plotting 2d t-Sne
x_axis = transformed[:, 0]
y_axis = transformed[:, 1]
if set_3d:
    z_axis = transformed[:, 2]
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_axis, y_axis, z_axis, c=clrs, s=plotsize)
else:
    ax2 = fig.add_subplot(122)
    ax2.scatter(x_axis, y_axis, c=clrs, s=plotsize)
#plt.subplot(1,2,2)
ax2.set_title("t-SNE")
#plt.scatter(x_axis, y_axis, c=labels)
#plt.scatter(x_axis, y_axis, z_axis, c=clrs)

#plt.legend(clrs, ['Cluster 1', 'Cluster 2', 'Noise'])

plt.show()
