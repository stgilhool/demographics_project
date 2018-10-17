import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Initialize paths and file names
demo_data_path = "/home/gilhools/demographics_project/"

seq_data_path = demo_data_path + "seq_data/"
seq_data_file = seq_data_path + "seq_data.csv"

patient_data_filename = "patient_demo_all.csv" #patient data
patient_data_file = demo_data_path + patient_data_filename

cag_survey_filename = "survey.csv" #CAG survey data
#cag_survey_filename = "test.csv" #CAG survey data
cag_survey_file = demo_data_path + cag_survey_filename


### Read in the CSV files
# Read in seq data file
seq_df = pd.read_csv(seq_data_file, dtype={'SAMPLE':str})
seq_df.rename({'SAMPLE':'SUBJECT_ID'}, axis=1, inplace=True)

# Get just the columns that are relevant
patient_df_cols = ['pat_id', 'pat_gender', 'pat_race', 'pat_ethnicity']
patient_df = pd.read_csv(patient_data_file, dtype={'pat_id':str},
                         usecols=patient_df_cols) #make sure id's are strings
patient_df.rename({'pat_id':'SUBJECT_ID'}, axis=1, inplace=True) #ID column name must match in both df's

survey_df_cols = ['SUBJECT_ID', 'FORMCODE', 'CHILDGENDER', 'RACE', 'ISASIAN',
                  'ASIANTYPE', 'SAMPLETYPE', 'HASMRN'] #get just the columns we want
survey_df = pd.read_csv(cag_survey_file, encoding='latin_1', low_memory=False,
                        dtype={'SUBJECT_ID':str}, usecols=survey_df_cols) #make sure subject_id is a string


#Merge seq table with epic patient table, keeping only the overlaps
seq_pat_df = pd.merge(seq_df, patient_df, on='SUBJECT_ID', how='inner')
seq_pat_df.shape


#id_list = survey_df.loc[:,'SUBJECT_ID']
#id_list_vals = [x for x in id_list]
#id_ndups = [id_list_vals.count(x) for x in id_list_vals]
#
#d_idx = np.where(id_list_vals == id_list_vals[7])
#
#
#print(d_idx)

#def dup_list(ids):
#
#    output_list = []
#    
#    for pat_id in ids:
#        idx = ids.index(pat_id)
#        output_list.append(idx)
#
#    return output_list
#
#duplications=dup_list(survey_df.loc[:,'SUBJECT_ID'])
#print(duplications.shape)

# Look at duplicate entries

# Merge the tables
#full_table = pd.merge(patient_df, survey_df, on='SUBJECT_ID')





# Read in seq data
# Merge patient IDs with seq data
# Do some magic
