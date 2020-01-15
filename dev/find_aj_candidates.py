import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances as pairwise
import cag_input_data_mm as mm

df = pd.read_pickle('/home/gilhools/demographics_project/data/patient_data/autoencoder_results.pkl')
df = df.loc[df.index.dropna()]

ajdf = df.loc[df.LABELS=='Ashkenazi']

aj_mean = ajdf.iloc[:,0:-1].mean(axis=0)

aj_std = ajdf.iloc[:,0:-1].std(axis=0)

nsigma = 3
c0_limits = aj_mean.values[0] + (nsigma * aj_std.values[0] * np.array([-1.,1.]))
c1_limits = aj_mean.values[1] + (nsigma * aj_std.values[1] * np.array([-1.,1.]))

aj_candidate_df = df.loc[df.apply(lambda x: (x[0] >= c0_limits[0]) and
                                  (x[0] <= c0_limits[1]) and
                                  (x[1] >= c1_limits[0]) and
                                  (x[1] <= c1_limits[1]), axis=1)]
# Get the AJ pats to the top of the df
aj_candidate_df = pd.concat([ajdf, aj_candidate_df])
aj_candidate_df.drop_duplicates(inplace=True)

distances = pairwise(aj_candidate_df.iloc[:,0:-1].values)

dist_df = pd.DataFrame(index=aj_candidate_df.index,
                       columns=aj_candidate_df.index,
                       data=distances)
