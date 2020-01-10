import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

from cag_input_data_mm import DataSet
from cag_input_data_mm import PATIENT_DATA_PATH

RESULTS_PATH = '/home/gilhools/demographics_project/out/'

lookup_table = pd.read_pickle(PATIENT_DATA_PATH+'sid_lookup.pkl')

# Read in ae and tsne results
ae_df = pd.read_pickle(RESULTS_PATH+'autoencoder_results.pkl')
tsne_df = pd.read_pickle(RESULTS_PATH+'tsne_df.pkl')

# Cut ae table to just Caucasian and Black/AA patients
ae_df = ae_df.loc[ae_df.LABELS.isin([
    "Caucasian",
    "Black/African American",
    "Other",
    #"Caucasian,Black/African American",
    #"Hispanic/Latino + Other",
    #"Hispanic/Latino + Caucasian",
    #"Hispanic/Latino + Black",
])]

pca_df = pd.read_csv('/home/gilhools/demographics_project/data/seq_data/MM-PCA.eigenvec', sep=' ', header=None)

pca_df = pca_df.loc[:,[1,2,3,4,5,6,7,8,9,10,11]]
pca_df['SUBJECT_ID'] = pca_df[1].map(lookup_table).values
pca_df.set_index('SUBJECT_ID',inplace=True)
pca_df.rename({2:'PCA_1', 3:'PCA_2', 4:'PCA_3', 5:'PCA_4', 6:'PCA_5', 7:'PCA_6', 8:'PCA_7', 9:'PCA_8', 10:'PCA_9', 11:'PCA_10'}, axis=1, inplace=True)


#change index of tsne_df from InNDIV_ID to SUBJECT_ID
tsne_df['SUBJECT_ID'] = tsne_df.reset_index()['INDIV_ID'].map(lookup_table).values
tsne_df.set_index('SUBJECT_ID', inplace=True)

#merge the fuckers
merge_df = pd.merge(ae_df, tsne_df, left_index=True, right_index=True, how='left')
merge_df = pd.merge(merge_df, pca_df, left_index=True, right_index=True, how='left')

def balance_classes(df):
    df_copy = df.copy()
    #nclasses = df.LABELS.value_counts().count()
    nsamples_per_class = df_copy.LABELS.value_counts().values
    classes = df_copy.LABELS.value_counts().index
    nsamples_max_class = max(nsamples_per_class)
    for cls, nsamples in list(zip(classes, nsamples_per_class)):
        nsamples_needed = nsamples_max_class - nsamples
        #add nsamples_needed randomly sampled to dataframe
        df_copy = pd.concat([df_copy, df_copy.loc[df_copy.LABELS == cls].sample(nsamples_needed, replace=True)])
    df_copy = df_copy.sample(frac=1)
    return df_copy

# To plot
#merge_df.plot.scatter(x='CODES_0', y='CODES_1', color=merge_df.LABELS.map({'Caucasian':'blue', 'Black/African American':'red'}), alpha=0.3)
#merge_df.plot.scatter(x='TSNE_0', y='TSNE_1', color=merge_df.LABELS.map({'Caucasian':'blue', 'Black/African American':'red'}), alpha=0.3)
def read_data_classifier(fake_data=False,
                         one_hot=False,
                         input_type='ae',
                         validation_size=450,
                         test_size=1000,
                         dtype=tf.float16):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    # Read in real data
    input_df = merge_df
    #input_df = balance_classes(input_df)

    if input_type == 'ae':
        input_data = input_df.loc[:,['CODES_0','CODES_1']].values
    elif input_type == 'tsne':
        input_data = input_df.loc[:,['TSNE_0','TSNE_1']].values
    elif input_type == 'pca':
        input_data = input_df.loc[:,['PCA_1','PCA_2','PCA_3','PCA_4','PCA_5','PCA_6','PCA_7','PCA_8','PCA_9','PCA_10']].values
    else:
        raise ValueError('Input type must be ae or tsne')
    
    input_labels_string_series =  input_df.LABELS
    input_labels_string = input_labels_string_series.values.reshape(-1,1)
    # Change labels to one-hot encodings
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(input_labels_string)
    input_labels = enc.transform(input_labels_string).toarray()

    print("input_labels.shape = ", input_labels.shape)
    test_images = input_data[:test_size,:]
    test_labels = input_labels[:test_size]
    print("test_labels.shape = ", test_labels.shape)
    
    trainval_images = input_data[test_size:,:]
    trainval_labels = input_labels[test_size:]
    # Take subset of training data for validation
    validation_images = trainval_images[:validation_size,:]
    validation_labels = trainval_labels[:validation_size]
    # And keep subset 
    train_images = trainval_images[validation_size:,:]
    train_labels = trainval_labels[validation_size:]
    # get rid of trainval
    del trainval_images
    del trainval_labels
    
    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(validation_images, validation_labels,
                                   dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
    return data_sets
