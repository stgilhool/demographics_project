from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import cag_input_data_mm
import numpy as np
import pandas as pd
import csv
import pdb
#from mpl_toolkits.mplot3d import Axes3D

def ancestry_tsne(df,
                verbose=False,
                **kwargs):
    '''
    Run t-SNE on pca data
    '''
    data = df.values
    df_index = df.index
    del df

    # t-SNE
    if verbose:
        print("Initializing t-SNE model... ", end='', flush=True)

    model = TSNE(n_components=kwargs.pop('n_components',2),
                 perplexity=kwargs.pop('perplexity',30),
                 learning_rate=kwargs.pop('learning_rate',100),
                 early_exaggeration=kwargs.pop('early_exaggeration',12.0),
                 init=kwargs.pop('init','pca'),
                 random_state=kwargs.pop('random_state',12345),
                 method=kwargs.pop('method','barnes_hut'),
                 **kwargs)
    if verbose:
        print("Done.")

        # Fitting Model
        print("Performing t-SNE... ", end='', flush=True)

    transformed = model.fit_transform(data)

    if verbose:
        print("Done.")

    tsne_df = pd.DataFrame(data=transformed,
                           index=df_index,
                           columns=['TSNE_0','TSNE_1'])
    return tsne_df

input_data = cag_input_data_mm.readin_pca(rescale=False)
tsne_df = ancestry_tsne(verbose=True)

print("Writing tsne df to disk")

tsne_df.to_pickle('tsne_df.pkl')
