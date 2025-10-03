import numpy as np
from glob import glob
from sklearn.preprocessing import OneHotEncoder

def prepare_dataset(feature_type, train_or_test = 'train', window_size = 0):
    # Loading the y data and obtaining some convenience variables
    if train_or_test.casefold() == 'train':
        y_data = np.loadtxt('../bit28390-sup-0001-supplementary_datafile_1.tsv', comments = None, dtype = object, skiprows = 1)
    else:
        y_data = np.loadtxt('../bit28390-sup-0003-supplementary_datafile_3.tsv', comments = None, dtype = object, skiprows = 1)
    valid_chromosomes = {'NW_023276807.1', 'NW_023276806.1', 'NC_048595.1', 'NC_048596.1', 'NC_048597.1', 'NC_048598.1',
                         'NC_048599.1', 'NC_048600.1', 'NC_048601.1', 'NC_048602.1', 'NC_048603.1', 'NC_048604.1'}
    valid_samples = [elem in valid_chromosomes for elem in y_data[:, -3]]
    y_data = y_data[valid_samples]
    pdl_36 = y_data[:, 1].astype(float) # These are ln-transformed
    pdl_72 = y_data[:, 2].astype(float)
    hotspot_size = 16
    # Generating the X data from the processed chromatin and/or methylation files
    if feature_type.casefold() in {'methylation', 'both'}:
        feature_type = feature_type.casefold() + f'-{window_size:02}window'
    if not glob(f'{train_or_test}_X_data_{feature_type}.csv') and 'both' not in feature_type.casefold(): # This step takes a few minutes to run, so it is ran only if the X_data file is not present
        # Listing the files holding the X data
        if feature_type.casefold() == 'chromatin':
            files_to_read = sorted(glob('*peaks.narrowPeak'))
        elif 'methylation' in feature_type.casefold():
            files_to_read = ['../CpG_OT_bed.tsv', '../CpG_OB_bed.tsv']
        X_data = np.zeros((y_data.shape[0], len(files_to_read) + (hotspot_size + 2*window_size - 2)*('methylation' in feature_type.casefold())  )) # hotspot_size + 2*window_size - 2 because two entries already comes from files_to_read
        # For each entry with a productivity, get all chromatin peaks at different types and timepoints, creating the X data matrix
        for file_idx, this_file in enumerate(files_to_read):
            if this_file.endswith('.narrowPeak'): # Chromatin
                chromatin_data = np.loadtxt(this_file, comments = None, dtype = object)
                chromatin_start_indices = chromatin_data[:, 1].astype(int) # The start of a chromatin peak
                chromatin_end_indices = chromatin_data[:, 2].astype(int)
                chromatin_values = chromatin_data[:, 6].astype(float)
                for idx in range(X_data.shape[0]):
                    if not idx % 500:
                        print(f'Current chromatin file: {file_idx+1}/{len(files_to_read)} | Current sample: {idx:5,}/{X_data.shape[0]:,}', end = '\r')
                    hotspot_location = int(y_data[idx, -2]) # The location of this hotspot in its chromosome
                    data_loc = (chromatin_data[:, 0] == y_data[idx, -3]) & (chromatin_start_indices <= hotspot_location) # Sample has to be: in the right chromosome & located after the beginning of a peak
                    if np.any(data_loc) and chromatin_end_indices[data_loc][-1] >= hotspot_location: # Checking whether sample is located before the end of a peak using only the last peak
                        X_data[idx, file_idx] = chromatin_values[data_loc][-1]
            else: # Methylation
                methylation_data = np.loadtxt(this_file, comments = None, dtype = object, delimiter = '\t', skiprows = 1)
                methylation_indices = methylation_data[:, 2].astype(int)
                methylation_values = methylation_data[:, 3].astype(float)
                if 'OT' in this_file: # Assumes that the top strand is represented by a '+' in the Hilliard and Lee data
                    strand_symbol = '+'
                else:
                    strand_symbol = '-'
                for idx in range(X_data.shape[0]):
                    if not idx % 200:
                        print(f'Current methylation file: {1 + int(strand_symbol == "-")}/2 | Current sample: {idx:5,}/{X_data.shape[0]:,}', end = '\r')
                    if strand_symbol != y_data[idx, -1]: # Some entries are for the other strand, so we can skip those
                        continue
                    hotspot_location = int(y_data[idx, -2]) # The location of this hotspot in its chromosome
                    data_loc = (methylation_data[:, 0] == y_data[idx, -3]) # Sample has to be: in the right chromosome
                    nearest_lower_methylation_idx = np.where(methylation_indices[data_loc] <= hotspot_location - window_size)[0][-1]
                    if hotspot_location - window_size != methylation_indices[data_loc][nearest_lower_methylation_idx] and (hotspot_location + hotspot_size + window_size < methylation_indices[data_loc][nearest_lower_methylation_idx+1]): # The nearest methylated site with a lower (or equal) idx is not the beginning of this hotspot, and the next methylation size is beyond this hotspot
                        continue
                    else:
                        if hotspot_location - window_size != methylation_indices[data_loc][nearest_lower_methylation_idx]:
                            nearest_lower_methylation_idx += 1 # A correction for convenience, as the nearest_lower_methylation_idx position may be for a methylation site before the hotspot, so it needs to be excluded
                        nearest_higher_methylation_idx = np.where(methylation_indices[data_loc] >= hotspot_location + hotspot_size + window_size)[0][0] # TODO: see whether a while loop is more efficient. Without while, took 1,193 secs
                        # nearest_higher_methylation_idx = nearest_lower_methylation_idx + 1
                        # while methylation_indices[data_loc][nearest_higher_methylation_idx] < hotspot_location + hotspot_size + window_size:
                        #     nearest_higher_methylation_idx += 1
                        positions_relative_to_window_start_location = methylation_indices[data_loc][nearest_lower_methylation_idx:nearest_higher_methylation_idx] - (hotspot_location - window_size)
                        X_data[idx, file_idx + positions_relative_to_window_start_location - 1*(strand_symbol == "-")] = methylation_values[data_loc][nearest_lower_methylation_idx:nearest_higher_methylation_idx] # -1 for the 2nd file to correct for the increase in file_idx
        np.savetxt(f'{train_or_test}_X_data_{feature_type}.csv', X_data, delimiter = ',', comments = '', fmt = '%g')
    elif not glob(f'{train_or_test}_X_data_{feature_type}.csv') and 'both' in feature_type.casefold(): # If feature_type is 'both', just merge the files from 'chromatin' and 'methylation-[number]window'
        chromatin_X_data = np.loadtxt(f'{train_or_test}_X_data_chromatin.csv', delimiter = ',', comments = None)
        methylation_X_data = np.loadtxt(f'{train_or_test}_X_data_methylation-{window_size:02}window.csv', delimiter = ',', comments = None)
        methylation_X_data[methylation_X_data>0] = 1
        methylation_X_data = methylation_X_data.sum(axis=1).reshape(-1, 1) # How many methylated residues exist in the entire window
        X_data = np.hstack((chromatin_X_data, methylation_X_data))
        np.savetxt(f'{train_or_test}_X_data_{feature_type}.csv', X_data, delimiter = ',', comments = '', fmt = '%g')
        OHE = OneHotEncoder()
        # One-hot encoded data is based off the train set, so we need to load the train set file if we are building the test dataset
        if train_or_test == 'test':
            methylation_X_data_train = np.loadtxt(f'train_X_data_methylation-{window_size:02}window.csv', delimiter = ',', comments = None)
            methylation_X_data_train[methylation_X_data_train>0] = 1
            methylation_X_data_train = methylation_X_data_train.sum(axis=1).reshape(-1, 1) # How many methylated residues exist in the entire window
            OHE.fit(methylation_X_data_train)
        else:
            OHE.fit(methylation_X_data)
        methylation_OHE = OHE.transform(methylation_X_data).toarray()[:, 1:] # Removing the "0 methylation" column as it is already represented by an all-0 vector (that is, avoiding multicollinearity)
        X_data_OHE = np.hstack((chromatin_X_data, methylation_OHE))
        np.savetxt(f'{train_or_test}_X_data_{feature_type}_OHE.csv', X_data_OHE, delimiter = ',', comments = '', fmt = '%g')
    else:
        X_data = np.loadtxt(f'{train_or_test}_X_data_{feature_type}.csv', delimiter = ',', comments = None) # Used only to print the number of all-0 samples in the X data
    # Saving the y data for regression
    # np.savetxt(f'{train_or_test}_data_pdl-36_raw-ln.csv', pdl_36, delimiter = ',', comments = '')
    # np.savetxt(f'{train_or_test}_data_pdl-72_raw-ln.csv', pdl_72, delimiter = ',', comments = '')
    # np.savetxt(f'{train_or_test}_data_pdl-36_exp.csv', np.exp(pdl_36), delimiter = ',', comments = '')
    # np.savetxt(f'{train_or_test}_data_pdl-72_exp.csv', np.exp(pdl_72), delimiter = ',', comments = '')
    # Saving the y data for classification
    if 'methylation' in feature_type.casefold():
        return # No need to recreate the y data, and we do not need the nonzero methylation file since it is never used
    elif 'chromatin' in feature_type.casefold():
        nonzero = np.any(X_data, axis = 1)
    else: # 'both'
        chromatin_X_data = np.loadtxt(f'{train_or_test}_X_data_chromatin.csv', delimiter = ',', comments = None)
        nonzero = np.any(chromatin_X_data, axis = 1)
    cl_data = np.exp(pdl_36); cl_data[cl_data>=0.7] = 1; cl_data[cl_data<0.7] = 0; cl_data = cl_data.astype(int) # Exponentiating to get the true values, assigning productivity >= 0.7 as the 1 (True) class and the rest as the 0 (False) class, and changing the data type to int
    cl_data72 = np.exp(pdl_72); cl_data72[cl_data72>=0.7] = 1; cl_data72[cl_data72<0.7] = 0; cl_data72 = cl_data72.astype(int)
    stable = cl_data & cl_data72
    np.savetxt(f'{train_or_test}_data_2class.csv', stable, delimiter = ',', comments = '', fmt = '%d')
    np.savetxt(f'{train_or_test}_data_2class_nonzero.csv', stable[nonzero], delimiter = ',', comments = '', fmt = '%d')
    stable[(np.exp(pdl_36)>=2)&(np.exp(pdl_72)>=2)] = 2 # Additional class added for highly productive samples
    np.savetxt(f'{train_or_test}_data_3class.csv', stable, delimiter = ',', comments = '', fmt = '%d')
    np.savetxt(f'{train_or_test}_data_3class_nonzero.csv', stable[nonzero], delimiter = ',', comments = '', fmt = '%d')
    print(f'Samples with all 0 {feature_type} X data in the {train_or_test} set: {(~nonzero).sum():,}/{X_data.shape[0]:,} ({100*(~nonzero).sum()/X_data.shape[0]:.2f}%)' + ' '*10) # ~np.any(X_data, axis = 1) could be np.all(X_data == 0, axis = 1)
    np.savetxt(f'{train_or_test}_X_data_{feature_type}_nonzero.csv', X_data[nonzero], delimiter = ',', comments = '', fmt = '%g')
    if 'X_data_OHE' in locals():
        np.savetxt(f'{train_or_test}_X_data_{feature_type}_OHE_nonzero.csv', X_data_OHE[nonzero], delimiter = ',', comments = '', fmt = '%g')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Prepares a dataset with the CHO epigenetic data (chromatin, methylation, or both) and the productivity data.')
    parser.add_argument('feature_type', type = str, nargs = 1, help = 'Whether to include only chromatin data, only methylation data, or both in the X data. Must be in {"chromatin", "methyltation", "both"}.')
    parser.add_argument('-ws', '--window_size', type = int, nargs = 1, metavar = 0, default = [0], help = 'The window size around a hotspot for which methylation data are included. Relevant only when feature_type is in {"methyltation", "both"}.')
    args = parser.parse_args()
    from time import time
    t1 = time()
    prepare_dataset(args.feature_type[0], 'train', args.window_size[0])
    prepare_dataset(args.feature_type[0], 'test', args.window_size[0])
    t2 = time()
    print(f'{t2-t1:.3f}     ')
