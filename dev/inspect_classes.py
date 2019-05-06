import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cag_input_data_mm as cdata

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
