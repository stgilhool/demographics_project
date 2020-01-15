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
'''
def scatter(codes, labels):
    colors = [
        ('#27ae60', 'o'),
        ('#2980b9', 'o'),
        ('#8e44ad', 'o'),
        ('#f39c12', 'o'),
        ('#c0392b', 'o'),
        ('#27ae60', 'x'),
        ('#2980b9', 'x'),
        ('#8e44ad', 'x'),
        ('#c0392b', 'x'),
        ('#f39c12', 'x'),
    ]
    for num in np.arange(8):
        plt.scatter([codes[:,0][i] for i in np.arange(len(labels)) if labels[i] == num],
        [codes[:,1][i] for i in np.arange(len(labels)) if labels[i] == num], 7,
        label=str(num), color = colors[num][0], marker=colors[num][1])
    plt.legend()
    plt.show()
'''

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
                                     '''Plot the output of the encoder
                                     in 2 or 3D''')
    parser.add_argument('savepath',
                        nargs=1,
                        type=str,
                        help="Path to tf checkpoint")

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
    parser.add_argument('-j','--aj','--ashkenazi',
                        action='store_true',
                        help='''
                        Option to overplot the Ashkenazi Jews
                        ''')
    parser.add_argument('-o', '--out', '--outfile',
                        dest='outfile',
                        type=str,
                        help='''
                        Filename for DataFrame output.
                        Will be stored in data directory as .pkl''')

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

    ajplot = args.aj
    outfile = args.outfile

    print("\nPULLING UP CAG DATA")
    '''
    cag = cag_input_data_mm.read_data_sets(input_type=input_type,
                                           n_data=n_snp,
                                           validation_size=VAL_SIZE,
                                           test_size=TEST_SIZE)
    print(len(cag.test.labels))
    print(cag.test.labels)
    test_images = cag.test.images
    test_labels = cag.test.labels
    '''

    cag = cag_input_data_mm.read_test_data()
    test_images = cag.images
    test_labels = cag.labels
    test_ids = cag.ids

    # print "\nSTARTING PCA"
    # pca = decomposition.PCA(n_components=2)
    # pca.fit(mnist.train.images)
    #
    # print "\nGENERATING PCA CODES AND RECONSTRUCTION"
    # pca_codes = pca.transform(mnist.test.images)
    # print pca_codes
    #
    # scatter(pca_codes, mnist.test.labels)

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, n_snp])
            # cag data image of shape n_row*n_col=n_snp

            phase_train = tf.placeholder(tf.bool)

            code = ae.encoder(x, n_code, phase_train)

            output = ae.decoder(code, n_code, phase_train)

            cost, train_summary_op = ae.loss(output, x)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = ae.training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(output, x)

            saver = tf.train.Saver()

            sess = tf.Session()


            print("\nSTARTING AUTOENCODER\n", args.savepath[0])

            saver.restore(sess, args.savepath[0])

            print("\nGENERATING AE CODES AND RECONSTRUCTION")
            '''
            ae_codes, ae_reconstruction = sess.run([code, output],
                                                   feed_dict=
                                                   {x: test_images *
                                                    np.random.randint(2,
                                                                      size=n_snp),
                                                    phase_train: True})
            '''
            ae_codes, ae_reconstruction = sess.run([code, output],
                                                   feed_dict=
                                                   {x: test_images,
                                                    phase_train: False})

            print(test_images.min(axis=0))
            print(test_images.max(axis=0))
            print(ae_reconstruction.min(axis=0))
            print(ae_reconstruction.min(axis=1))

            #Sort codes and labels by most frequent labels
            counter = Counter(test_labels)
            if n_code == 2:
                df_base = pd.DataFrame({'CODES_0':ae_codes[:,0],
                                        'CODES_1':ae_codes[:,1],
                                        'LABELS':test_labels})
            elif n_code == 3:
                df_base = pd.DataFrame({'CODES_0':ae_codes[:,0],
                                        'CODES_1':ae_codes[:,1],
                                        'CODES_2':ae_codes[:,2],
                                        'LABELS':test_labels})
            else:
                raise ValueError("n_code must be 2 or 3. Currently = {}".format(n_code))
            '''
            df_sorted = pd.DataFrame([])
            for lab, _ in counter.most_common():
                tmpdf = df_base.loc[df_base.LABELS == lab]
                df_sorted = pd.concat([df_sorted, tmpdf])
            del df_base
            del tmpdf
            '''
            df_sorted = df_base.set_index(test_ids)

            if n_code == 2:
                ae_codes = df_sorted.loc[:,['CODES_0','CODES_1']].values
            elif n_code == 3:
                ae_codes = df_sorted.loc[:,['CODES_0','CODES_1','CODES_2']].values
            else:
                raise ValueError("n_code must be 2 or 3")

            labels = df_sorted.LABELS.values

            #Writer = MovieWriter()
            #writer = Writer(fps=30, metadata=dict(artist='Gilhool'), bitrate=1800)

            metadata = dict(title='test_movie', artist='Gilhool',
                            comment='Test')
            fig = plt.figure(figsize=(14,9))

            if ae_codes.shape[1] == 2:
                ndim = 2
                ax = fig.add_subplot(111)
            elif ae_codes.shape[1] == 3:
                ndim = 3
                ax = fig.add_subplot(111, projection='3d')
            else:
                raise ValueError("Codes must have 2 or 3 dimensions")

            aj_idx = cag_input_data_mm.get_ashkenazi_idx().values
            df_sorted.loc[aj_idx,'LABELS']='Ashkenazi'
            if outfile:
                df_sorted.to_pickle('/home/gilhools/demographics_project/data/patient_data/'+outfile+'.pkl')
            if ajplot:
                scatter_aj(ae_codes, labels, aj_idx)
            else:
                scatter(ae_codes, labels)
            plt.legend()


            if ae_codes.shape[1] == 2:
                plt.show()
            elif ae_codes.shape[1] == 3:
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
