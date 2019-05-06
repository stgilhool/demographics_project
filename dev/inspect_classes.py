import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cag_input_data_mm import DataSet

RESULTS_PATH = '/home/gilhools/demographics_project/out/'

lookup_table = pd.read_pickle(PATIENT_DATA_PATH+'sid_lookup.pkl')

# Read in ae and tsne results
ae_df = pd.read_pickle(RESULTS_PATH+'autoencoder_results.pkl')
tsne_df = pd.read_pickle(RESULTS_PATH+'tsne_df.pkl')

# Cut ae table to just Caucasian and Black/AA patients
ae_df = ae_df.loc[ae_df.LABELS.isin(["Caucasian", "Black/African American"])]

#change index of tsne_df from InNDIV_ID to SUBJECT_ID
tsne_df['SUBJECT_ID'] = tsne_df.reset_index()['INDIV_ID'].map(lookup_table).values
tsne_df.set_index('SUBJECT_ID', inplace=True)

#merge the fuckers
merge_df = pd.merge(ae_df, tsne_df, left_index=True, right_index=True, how='left')

# To plot
#merge_df.plot.scatter(x='CODES_0', y='CODES_1', color=merge_df.LABELS.map({'Caucasian':'blue', 'Black/African American':'red'}), alpha=0.3)
#merge_df.plot.scatter(x='TSNE_0', y='TSNE_1', color=merge_df.LABELS.map({'Caucasian':'blue', 'Black/African American':'red'}), alpha=0.3)
def read_data_classifier(fake_data=False,
                         one_hot=False,
                         input_type='raw',
                         n_data=10,
                         validation_size=450,
                         test_size=1000,
                         dtype=tf.float32):
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
    input_df = merge_df()
    input_data = input_df.loc[:,['CODES_0','CODES_1']].values
    
    input_labels =  input_df.LABELS


    test_images = input_data[:test_size,:]
    test_labels = input_labels[:test_size]

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
