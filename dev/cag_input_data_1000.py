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
SEQ_DATA_PATH = DEMO_DATA_PATH + "seq_data/"
PATIENT_DATA_PATH = DEMO_DATA_PATH + "patient_data/"
THOU_DATA_PATH = DEMO_DATA_PATH + "1000_Genomes/"

# Filenames
PATIENT_DATA_FILENAME = "integrated_call_samples_v3.20130502.ALL.panel"
PCA_DATA_FILENAME = "pca_1000G.eigenvec"

# Full paths to files.
PATIENT_DATA_FILE = PATIENT_DATA_PATH + PATIENT_DATA_FILENAME
PCA_DATA_FILE = SEQ_DATA_PATH + PCA_DATA_FILENAME
THOU_DATA_FILE = THOU_DATA_PATH + PATIENT_DATA_FILENAME

def readin_pca(ncols=10, rescale=True):
    pca_cols = [#'UNKNOWN',
                #'INDIV_ID',
        'SUBJECT_ID',
                'PCA_0',
                'PCA_1',
                'PCA_2',
                'PCA_3',
                'PCA_4',
                'PCA_5',
                'PCA_6',
                'PCA_7',
                'PCA_8',
                'PCA_9'
                ]
    pca_df = pd.read_csv(PCA_DATA_FILE, header=None, names=pca_cols, sep=' ')
    pca_df.set_index('SUBJECT_ID', inplace=True)
    #rescale
    if rescale:
        #Depends upon all data being pca values and the IDs as index
        vals = pca_df.values
        newvals = (vals-vals.min())/(vals-vals.min()).max()
        pca_df = pd.DataFrame(data=newvals,
                              columns=pca_df.columns,
                              index=pca_df.index)
    
    pca_df = pca_df.iloc[:,np.arange(ncols)]
    return pca_df

def readin_1000G_labels():
    genomes_df = pd.read_csv(THOU_DATA_FILE,
                             header=0,
                             encoding='latin1',
                             sep='\t',
                             )
    genomes_df.rename(columns={'sample':'SUBJECT_ID', 'super_pop':'RACE'}, inplace=True)
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
    df = readin_1000G_labels()
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

def read_data_sets(input_type='raw',
                   n_data=10,
                   validation_size=450,
                   test_size=1000,
                   dtype=tf.float16):
    class DataSets(object):
        pass
    data_sets = DataSets()
    # Read in real data
    input_df = get_data_and_labels()
    input_data = input_df.iloc[:,0:10].values #FIXME: dependent on input_df columns (too hardcoded)
    input_labels =  input_df.RACE #FIXME: ? Why is this race and not LABELS?

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


def get_ashkenazi_idx(df=get_data_and_labels(),
                      filepath=PATIENT_DATA_PATH+'aj_data.txt'):
    ids = [line[0:10] for line in open(filepath,'r').readlines()]
    #bool_idx = df['SUBJECT_ID'].isin(ids)
    bool_idx = df.index.isin(ids)
    return bool_idx

def read_test_data():

    input_df = get_data_and_labels()
    input_data = input_df.iloc[:,0:10].values
    input_labels =  input_df.RACE.values
    #input_id = input_df.SUBJECT_ID.values
    #input_id = input_df.SUBJECT_ID.values
    input_id = input_df.index.values
    data_set = DataSet(input_data,input_labels,ids=input_id)

    return data_set

