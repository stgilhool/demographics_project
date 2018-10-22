import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import pdb

# Initialize paths and file names
# Paths
DEMO_DATA_PATH = "/home/gilhools/demographics_project/data/"
SEQ_DATA_PATH = DEMO_DATA_PATH + "seq_data/"
PATIENT_DATA_PATH = DEMO_DATA_PATH + "patient_data/"

# Filenames
PATIENT_DATA_FILENAME = "patient_demo_all.csv" 
CHIP_DATA_FILENAME = "chip_data.csv"
PED_DATA_FILENAME = "GSA-comb0-filt.ped"
MAP_DATA_FILENAME = "GSA-comb0-filt.map"

# Full paths to files.
PED_DATA_FILE = SEQ_DATA_PATH + PED_DATA_FILENAME
MAP_DATA_FILE = SEQ_DATA_PATH + MAP_DATA_FILENAME
CHIP_DATA_FILE = SEQ_DATA_PATH + CHIP_DATA_FILENAME
PATIENT_DATA_FILE = PATIENT_DATA_PATH + PATIENT_DATA_FILENAME

n_row = 10
n_col = 10
n_snp = n_row * n_col

def readin_epic():
    # Read in patient data file
    patient_df_cols = ['pat_id', 'pat_gender', 'pat_race', 'pat_ethnicity']
    patient_df = pd.read_csv(PATIENT_DATA_FILE, dtype={'pat_id':str},
                             usecols=patient_df_cols)    
    patient_df.set_index('pat_id', inplace=True)
    return patient_df

def readin_chip():
    # Read in chip data file.
    chip_df = pd.read_csv(CHIP_DATA_FILE, dtype={'SAMPLE':str})
    chip_df.rename({'SAMPLE':'pat_id'}, axis=1, inplace=True)
    chip_df.set_index('pat_id', inplace=True)
    return chip_df

def readin_map():
    print("Reading MAP file...", end='', flush=True)
    map_colnames = ['CHR',
                    'SNP_RS',
                    'POS_MG',
                    'POS_BP'
    ]

    map_df = pd.read_csv(MAP_DATA_FILE, header=0, names=map_colnames, sep='\t')
    print(" Done")
    return map_df

def readin_ped_old(snp_names, n_snp=n_snp):
    """ Readin ped file, adding columns names using the SNP names from
    the map file (<SNP1>_A1, <SNP1>_A2, <SNP2>_A1, <SNP2>_A2, ...),
    and allowing the user to choose the number of SNPs to read in
    """
    # FIXME: Check that n_snp is in range
    # FIXME: Check that snp_names is string array
    print("Reading PED file...", end='', flush=True)
    
    ped_colnames = ['FAM_ID',
                    'INDIV_ID',
                    'PAT_ID',
                    'MAT_ID',
                    'SEX',
                    'PHENOTYPE'
    ] # Plus genotype columns that we will add now
    
    # Get number of SNPs from map file
    n_snp_total = len(snp_names)

    # Make array of column names
    alleles = np.array(['_A1','_A2'])
    snp_names = np.array(snp_names, dtype=str)
    new_names = np.char.add(snp_names[0:n_snp, np.newaxis],
                            alleles).reshape(n_snp*2)

    ped_colnames = np.append(ped_colnames, new_names)

    # Make vector of indices for which columns we want to use
    col_idx_init = np.array([1,5])
    col_idx = np.append(col_idx_init, np.arange(n_snp*2)+6)

    # Get INDIV_ID, phenotype and (some) genotypes from ped file
    ped_df = pd.read_csv(PED_DATA_FILE, header=None,
                         usecols=col_idx, sep=' ')
    # Append the column names to the data frame
    ped_df.columns = ped_colnames[col_idx]
    print(" Done")

    return ped_df

def merge_snps(geno_arr):
    # Take array of snp genotype data (2 cols per snp) and merge
    # into one column consisting of a base-3 representation
    # 0 0 -> 0
    # 1 1 -> 4
    # 1 2 -> 5
    # 2 1 -> 7
    # 2 2 -> 8
    n_rows = geno_arr.shape[0]
    n_cols = geno_arr.shape[1]
    n_snp = int(n_cols/2)
    geno_arr = geno_arr.reshape(n_snp*n_rows, 2)

    base3_geno_arr = geno_arr * np.array([3**1, 3**0])
    base3_geno_arr = np.add.reduce(base3_geno_arr, axis=1)
    base3_geno_arr = base3_geno_arr.reshape(n_rows, n_snp)

    return base3_geno_arr



def readin_ped(snp_names, n_snp=n_snp):
    """ Readin ped file, adding columns names using the SNP names from
    the map file, and converting the genotype (2 alleles) into a single code,
    and allowing the user to choose the number of SNPs to read in
    """
    # FIXME: Check that n_snp is in range
    # FIXME: Check that snp_names is string array
    print("Reading PED file...", end='', flush=True)
    
    ped_colnames = np.array(['FAM_ID',
                             'INDIV_ID',
                             'PAT_ID',
                             'MAT_ID',
                             'SEX',
                             'PHENOTYPE'
    ]) # Plus genotype columns that we will add now
    
    # Get number of SNPs from map file
    n_snp_total = len(snp_names)

    # Make array of column names
    snp_names_all = np.array(snp_names, dtype=str)
    snp_names = snp_names_all[:n_snp]

    # PED file column names and indices that we want
    relevant_idx = np.array([1,5])
    ped_colnames = np.append(ped_colnames[relevant_idx], np.array(snp_names))
    ped_col_idx = np.append(relevant_idx,np.arange(n_snp*2)+6)
    

    # Get (some) genotypes from ped file
    ped_df = pd.read_csv(PED_DATA_FILE, header=None,
                         usecols=ped_col_idx, sep=' ')

    # Merge each of the genotype pairs into a single encoding per snp
    geno_arr = ped_df.iloc[:,len(relevant_idx):].values
    ped_df.drop(ped_df.columns[np.arange(n_snp*2)+len(relevant_idx)],
                axis=1, inplace=True)
    base3_geno_arr = merge_snps(geno_arr)    
    del geno_arr
        
    # Merge the original df with the new genotype data
    geno_df = pd.DataFrame(data=base3_geno_arr)
    ped_df = pd.concat([ped_df, geno_df], axis=1)
    print(ped_df.shape)
    
    # Append the column names to the data frame
    ped_df.columns = ped_colnames
    print(f" Done ({n_snp} SNPS read)")

    return ped_df


#if __name__ == '__main__':

# Read in the data frames
patient_df = readin_epic()
chip_df = readin_chip()
map_df  = readin_map()

snp_names = map_df['SNP_RS'].values
ped_df = readin_ped(snp_names)




# Merge CHIPID and CHIP_SECTION to create INDIV_ID
chip_df['INDIV_ID'] = chip_df['CHIPID'].map(str) + '_' + chip_df['CHIP_SECTION']
# Pull out just the INDIV_ID from chip_df
indiv_lookup = pd.DataFrame(chip_df['INDIV_ID'])
# Concatenate (axis=1) the INDIV_ID onto the patient_df
pat_df = patient_df.join(indiv_lookup, how='inner')

# Merge the plink files and the other dataframe on the INDIV_ID.
data_df = pat_df.join(ped_df.set_index('INDIV_ID'), on='INDIV_ID', how='inner')

# Now separate into labels and data
data_split = np.split(data_df, [5], axis=1)
labels_df = data_split[0]
newdata_df = data_split[1]
indiv_id = labels_df.pop('INDIV_ID')

race_to_bit = {'ASIAN':(1<<1),
               'BLACK OR AFRICAN AMERICAN':(1<<2),
               'WHITE':(1<<3),
               'OTHER':(1<<4)}

bit_to_race = {value:key for key, value in race_to_bit.items()}

ethnicity_to_bit = {'HISPANIC':1,
                    'NON-HISPANIC':0}

bit_to_ethnicity = {value:key for key, value in ethnicity_to_bit.items()}

gd_idx = labels_df['pat_race'].notnull()


demographic_cols = labels_df.loc[gd_idx,['pat_race','pat_ethnicity']].values
#race_names = labels_df.loc[labels_df['pat_race'].notnull(),'pat_race'].values
race_names = demographic_cols[:,0]
ethnicity_names = demographic_cols[:,1]

race_embedding = [race_to_bit[race_names[i]] for i in range(len(race_names))]
ethnicity_embedding = [ethnicity_to_bit[ethnicity_names[i]]
                       for i in range(len(race_names))]

# 1-D array of labels
input_labels = np.array(race_embedding) + np.array(ethnicity_embedding)

# Genotype data as 2-D array (n_patients x n_genotypes)
input_data = newdata_df.iloc[:,np.arange(n_row*n_col)].values
input_data = input_data[gd_idx,:]

# Find out the "missingness" (frequency of 0's) per SNP
num_false = np.add.reduce(np.equal(input_data, 0), axis=0)
patients_total = float(input_data.shape[0])

percentage_missing = num_false/patients_total
# Above not needed so far (few data are missing)

print(labels_df.columns.values)
print(newdata_df.columns.values[0:10])

print("Number of entries in pat_df: {}".format(len(pat_df['INDIV_ID'].values)))
print("Number of entries in data_df: {}".format(len(data_df['INDIV_ID'].values)))

print(f'Shape of input data: {input_data.shape}')
print(f'Shape of input labels: {input_labels.shape}')


class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.uint8):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be 
    `uint8` 
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype != tf.uint8:
      raise TypeError('Invalid image dtype %r, expected uint8' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * n_snp
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
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
#def read_data_sets(input_data, input_labels, fake_data=False,
def read_data_sets(fake_data=False,
                   one_hot=False, dtype=tf.uint8):
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
  VALIDATION_SIZE = 450
  train_images = input_data[0:VALIDATION_SIZE*2,:]
  train_labels = input_labels[0:VALIDATION_SIZE*2]
  test_images = input_data[VALIDATION_SIZE*2:,:]
  test_labels = input_labels[VALIDATION_SIZE*2:]
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
  return data_sets
'''
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
'''
