import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import pdb
from sklearn import preprocessing

# Initialize paths and file names
# Paths
DEMO_DATA_PATH = "/home/gilhools/demographics_project/data/"
DATASET_PATH = DEMO_DATA_PATH + "adhd_xiao_gsa/"

PCA_DATA_PATH = DATASET_PATH + "pca_data/"
LABEL_DATA_PATH = DATASET_PATH + "label_data/"

# Filenames
LABEL_DATA_FILENAME = "adhd_gsa_all_labels.txt"
PCA_DATA_FILENAME = "adhd_gsa_all.evec"

# Full paths to files.
LABEL_DATA_FILE = LABEL_DATA_PATH + LABEL_DATA_FILENAME
PCA_DATA_FILE = PCA_DATA_PATH + PCA_DATA_FILENAME

def readin_pca(ncols=10, rescale=True):
    '''
    Reads in PCA_DATA_FILE into a dataframe.
    File must be tab-separated, with header.
    Columns are SUBJECT_ID, followed by PCA vectors.
    File must have headers, but columns will be renamed, so it
    doesn't matter what the names are in the original file

    INPUTS:
    ncols - number of PCA vectors (default 10)
    rescale - boolean (True). Rescale the vectors to range [0,1]

    OUTPUT:
    pca_df - DataFrame with SUBJECT_IDs as the index, and PCAs as the data
    '''
    pca_cols = ['SUBJECT_ID'] + ["_".join(["PCA",str(x)]) for x in np.arange(ncols)]
    pca_df = pd.read_csv(PCA_DATA_FILE, header=0, names=pca_cols, delim_whitespace=True)
    pca_df.set_index('SUBJECT_ID', inplace=True)
    #rescale
    if rescale:
        #Depends upon all data being pca values and the IDs as index
        vals = pca_df.values
        newvals = (vals-vals.min())/(vals-vals.min()).max()
        pca_df = pd.DataFrame(data=newvals,
                              columns=pca_df.columns,
                              index=pca_df.index)
    return pca_df

def readin_labels():
    genomes_df = pd.read_csv(LABEL_DATA_FILE,
                             header=0,
                             encoding='latin1',
                             sep='\t',
                             )
    return genomes_df


def encode_labels(labels):
    '''Convert RACE strings into numerical codes'''
    le = preprocessing.LabelEncoder()
    le.fit(list(labels))
    encoding = le.fit_transform(list(labels))
    return encoding

def add_encoding(df, col='RACE'):
    '''Convert and then add numerical encoding for labels to df'''
    labels = df[col].values
    encoding = encode_labels(labels)
    df['LABELS'] = encoding
    return df


# Read in the data frames
def get_data_and_labels(ncols=10, rescale=True):
    pca = readin_pca(ncols=ncols, rescale=rescale)
    df = readin_labels()
    df.drop_duplicates('SUBJECT_ID', inplace=True) #FIXME: No dups in 1000G data, but maybe in other label sets

    df.set_index('SUBJECT_ID', inplace=True)

    df = add_encoding(df)
    out = pca.join(df, how='left')
    fillna_vals = {'LABELS':-1, 'RACE':'NA'}
    out.fillna(value=fillna_vals, inplace=True)
    #FIXME: There was a dropna step here, but I don't think it's necessary for this data set
    return out

class DataSet(object):
    def __init__(self, images, labels,
                 ids=None,
                 dtype=tf.float16):
        """Construct a DataSet.
        """
        # dtype = tf.as_dtype(dtype).base_dtype #FIXME: I think this is unneeded. It maybe coerces dtypes that are not tf.Dtypes?

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._ids = ids
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def ids(self):
        return self._ids
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(n_data=10,
                   col="LABELS",
                   validation_size=450,
                   test_size=1000,
                   dtype=tf.float16):
    class DataSets(object):
        pass
    data_sets = DataSets()
    # Read in real data
    input_df = get_data_and_labels(ncols=n_data)
    input_data = input_df.iloc[:,0:-2].values #FIXME: relies upon last two columns being Race and Labels
    input_labels =  input_df[col].values #FIXME: ? Why was this race and not LABELS?

    # Hold out test data
    test_images = input_data[:test_size,:]
    test_labels = input_labels[:test_size]
    # Training data will be split into training set and validation set
    train_images = input_data[test_size:,:]
    train_labels = input_labels[test_size:]
    # define validation set
    validation_images = train_images[:validation_size,:]
    validation_labels = train_labels[:validation_size]
    # redefine training set to exlude validation set
    train_images = train_images[validation_size:,:]
    train_labels = train_labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(validation_images, validation_labels,
                                   dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
    return data_sets

def read_test_data(col="LABELS"):

    input_df = get_data_and_labels()
    input_data = input_df.iloc[:,0:-2].values
    input_labels =  input_df[col].values
    input_id = input_df.index.values
    data_set = DataSet(input_data,input_labels,ids=input_id)

    return data_set

