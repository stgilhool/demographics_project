from sklearn import decomposition
from matplotlib import pyplot as plt
import tensorflow as tf
import autoencoder_cag as ae
import argparse, cag_input_data_mm
import numpy as np
from collections import Counter
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
#import seaborn as sns
# model-checkpoint-1001-329329

n_row = 2
n_col = 5
n_snp = n_row * n_col
input_type = 'pca'

VAL_SIZE = 1000
TEST_SIZE = 30000

def animate(angle):
    ax.azim += 2

def scatter(codes, labels, top_n=8):

    #fig = plt.figure()
    '''
    if codes.shape[1] == 2:
        ndim = 2
        ax = fig.add_subplot(111)
    elif codes.shape[1] == 3:
        ndim = 3
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("Codes must have 2 or 3 dimensions")
    '''
    ndim = codes.shape[1]
    plotsize = 7
    counter = Counter(labels)
    if top_n <= len(set(labels)):
        nlabels = top_n
    else:
        nlabels = len(set(labels))
    colors = cm.tab10(np.linspace(0,1,nlabels))
    for index, element in enumerate(counter.most_common()):
        if index < nlabels and element[0] != 'NA':
            lab = element[0]
            x_vals = [codes[:,0][i] for i in np.arange(len(labels)) if labels[i] == lab]
            y_vals = [codes[:,1][i] for i in np.arange(len(labels)) if labels[i] == lab]
            if ndim == 2:
                ax.scatter(x_vals,
                           y_vals,
                           s=plotsize,
                           label=str(lab), color = colors[index], marker='o', alpha=0.5)
            elif ndim == 3:
                z_vals = [codes[:,2][i] for i in np.arange(len(labels)) if labels[i] == lab]
                ax.scatter(x_vals,
                           y_vals,
                           z_vals,
                           s=plotsize,
                           label=str(lab), color = colors[index], marker='o', alpha=0.6)

def scatter_aj(codes, labels, aj_idx):

    plotsize = 7
    counter = Counter(labels)

    nlabels = 8
    print(codes.shape)
    print(labels.shape)
    print(aj_idx.shape)
    colors = cm.tab10(np.linspace(0,1,8))
    for index, element in enumerate(counter.most_common()):
        if index < nlabels and element[0] != 'NA':
            lab = element[0]
            x_vals = [codes[:,0][i] for i in np.arange(len(labels)) if labels[i] == lab]
            y_vals = [codes[:,1][i] for i in np.arange(len(labels)) if labels[i] == lab]
            if ndim == 2:
                ax.scatter(x_vals,
                           y_vals,
                           s=plotsize,
                           label=str(lab), color = colors[index], marker='o', alpha=0.5)
            elif ndim == 3:
                z_vals = [codes[:,2][i] for i in np.arange(len(labels)) if labels[i] == lab]
                ax.scatter(x_vals,
                           y_vals,
                           z_vals,
                           s=plotsize,
                           label=str(lab), color = colors[index], marker='o', alpha=0.2)
    #Plot Ashkenazi Jews
    x_vals = codes[:,0][aj_idx]
    y_vals = codes[:,1][aj_idx]
    if ndim == 2:
        ax.scatter(x_vals,
                   y_vals,
                   s=100,
                   label='Ashkenazi Jew', color = 'red', marker='P')
    elif ndim == 3:
        z_vals = codes[:,2][aj_idx]
        ax.scatter(x_vals,
                   y_vals,
                   z_vals,
                   s=100,
                   label='Ashkenazi Jew', color = 'red', marker='P')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     '''Plot the output of the tine or pca
                                     in 2 or 3D''')
    parser.add_argument('--mode', '-m',
                        choices=['tsne','pca'],
                        type=str,
                        default='pca',
                        help="tsne or pca mode")

    parser.add_argument('--n_code', '-n',
                        type=int,
                        choices=[2,3],
                        default=2,
                        help="Number of nodes in compressed representation")

    parser.add_argument('-a','--animate',
                        type=str,
                        nargs='?',
                        const=' ',
                        default=None,
                        help='''
                        Option to animate 3d plot. If no file name specified,
                        animation is shown on screen. If file name given,
                        animation will be written to disk''')
    args = parser.parse_args()

    n_code = args.n_code

    show_animation = False
    write_animation = False
    anim_filename = ''
    if args.animate:
        if args.animate == ' ':
            show_animation = True
        else:
            write_animation = True
            anim_filename = args.animate

    print("\nPULLING UP CAG DATA")

    cag = cag_input_data_mm.read_test_data()
    test_images = cag.images
    test_labels = cag.labels
    test_ids = cag.ids
    TSNE_2D_FILE = '/home/gilhools/demographics_project/out/tsne_df.pkl'
    TSNE_3D_FILE = '/home/gilhools/demographics_project/out/tsne_df_3d.pkl'

    if args.mode == 'pca':
        latent_data = test_images[:,0:n_code]
    elif args.mode == 'tsne':
        if n_code == 2:
            tsne_df = pd.read_pickle(TSNE_2D_FILE)
            latent_data = tsne_df.loc[:,['TSNE_0','TSNE_1']].values
        elif n_code == 3:
            tsne_df = pd.read_pickle(TSNE_3D_FILE)
            latent_data = tsne_df.loc[:,['TSNE_0','TSNE_1','TSNE_2']].values
        else:
            raise ValueError("n_code must be 2 or 3")

    labels = test_labels

    #Writer = MovieWriter()
    #writer = Writer(fps=30, metadata=dict(artist='Gilhool'), bitrate=1800)

    metadata = dict(title='test_movie', artist='Gilhool',
                    comment='Test')
    fig = plt.figure(figsize=(14,9))

    if latent_data.shape[1] == 2:
        ndim = 2
        ax = fig.add_subplot(111)
    elif latent_data.shape[1] == 3:
        ndim = 3
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("Codes must have 2 or 3 dimensions")

    #FIXME: Hardcoding AJ plot
    ajplot = True
    if ajplot:
        aj_idx = cag_input_data_mm.get_ashkenazi_idx().values
        scatter_aj(latent_data, labels, aj_idx)
    else:
        scatter(latent_data, labels)
    plt.legend()


    if latent_data.shape[1] == 2:
        plt.show()
    elif latent_data.shape[1] == 3:
        if write_animation:
            writer = animation.FFMpegWriter(fps=30, metadata=metadata)
            with writer.saving(fig, anim_filename+'.mp4', dpi=100):
                for j in range(180):
                    animate(j)
                    writer.grab_frame()
        elif show_animation:
            ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=180,
                                  repeat=True)
            plt.show()
        else:
            plt.show()
    else:
        raise ValueError("Codes must have 2 or 3 dimensions")
