from sklearn import decomposition
from matplotlib import pyplot as plt
import tensorflow as tf
import autoencoder_cag as ae
import argparse, cag_input_data_mm
import numpy as np
from collections import Counter
import matplotlib.cm as cm
import pandas as pd
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

def scatter(codes, labels, top_n=8):

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
            plt.scatter([codes[:,0][i] for i in np.arange(len(labels)) if labels[i] == lab],
                        [codes[:,1][i] for i in np.arange(len(labels)) if labels[i] == lab],
                        s=plotsize,
                        label=str(lab), color = colors[index], marker='o')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various optimization strategies')
    parser.add_argument('savepath', nargs=1, type=str)
    args = parser.parse_args()

    print("\nPULLING UP CAG DATA")
    cag = cag_input_data_mm.read_data_sets(input_type=input_type,
                                           n_data=n_snp,
                                           validation_size=VAL_SIZE,
                                           test_size=TEST_SIZE)
    print(len(cag.test.labels))
    print(cag.test.labels)

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

            code = ae.encoder(x, 2, phase_train)

            output = ae.decoder(code, 2, phase_train)

            cost, train_summary_op = ae.loss(output, x)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = ae.training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(output, x)

            saver = tf.train.Saver()

            sess = tf.Session()


            print("\nSTARTING AUTOENCODER\n", args.savepath[0])

            saver.restore(sess, args.savepath[0])

            print("\nGENERATING AE CODES AND RECONSTRUCTION")
            ae_codes, ae_reconstruction = sess.run([code, output], feed_dict={x: cag.test.images * np.random.randint(2, size=(n_snp)), phase_train: True})

            #Sort codes and labels by most frequent labels
            counter = Counter(cag.test.labels)
            df_base = pd.DataFrame({'CODES_0':ae_codes[:,0],
                                    'CODES_1':ae_codes[:,1],
                                    'LABELS':cag.test.labels})
            df_sorted = pd.DataFrame([])
            for lab, _ in counter.most_common():
                tmpdf = df_base.loc[df_base.LABELS == lab]
                df_sorted = pd.concat([df_sorted, tmpdf])
            del df_base
            del tmpdf

            ae_codes = df_sorted.loc[:,['CODES_0','CODES_1']].values
            labels = df_sorted.LABELS.values

            scatter(ae_codes, labels)

            #plt.imshow(ae_reconstruction[0].reshape((28,28)), cmap=plt.cm.gray)
            plt.show()
