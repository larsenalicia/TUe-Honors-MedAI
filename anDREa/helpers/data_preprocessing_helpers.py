import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal
from scipy.fft import fftshift
import math
import mne
from IPython import get_ipython
import re
from collections import OrderedDict
from statistics import mean, median 
import random
import scipy
import seaborn
import scipy.ndimage as sp
import tqdm as tqdm
from PIL import Image 


# Combine datasets _Sdb and _l

def get_all_users(dataset_dir) -> dict:
    """Returns a dictionary with user-IDs as keys and empty values.
    """
    user_dict = {}
    pattern = '\d+'
    for file in os.listdir(dataset_dir):
        try: user_id = int(re.findall(pattern, file)[0])
        except: pass
        user_dict[user_id] = ''
    return dict(OrderedDict(sorted(user_dict.items())))

def column_renaming():
    """Used to rename SDB dataframes in function 'combine_sdb_l'.
    """
    old_columns = [i for i in range(0, 100, 1)]
    new_columns = [i/2 for i in range(0, 100, 1)]
    return {old_columns[i]: new_columns[i] for i in range(0, 100)}

def combine_sdb_l(dataset_dir, user_id, change_columns):
    """Concatenate SDB and consciousness state data, per user.
    """
    df_sdb = pd.read_csv(f'{dataset_dir}/{user_id}_Sdb.csv', header=None).T.rename(columns=change_columns)
    df_l = pd.read_csv(f'{dataset_dir}/{user_id}_l.csv', header=None, names=['state'])
    df_data = pd.concat([df_sdb, df_l], axis=1)
    return df_data

def separate_consciousness_state(dataset_dir, user_id, change_columns):
    """Separate data depending on consciousness_state, per user.
    """
    df_data = combine_sdb_l(dataset_dir, user_id, change_columns)
    df_unconsciousness = df_data[df_data['state'] == 0].reset_index(drop=True).drop(labels='state', axis=1)
    df_consciousness = df_data[df_data['state'] == 1].reset_index(drop=True).drop(labels='state', axis=1)
    return df_unconsciousness, df_consciousness

def store_data_per_user(dataset_dir, change_columns):
    """Store all the concatenated data as values in a dictionary with user-IDs as keys.
    """
    user_dict = get_all_users(dataset_dir)
    for user_id in user_dict.keys():
        df_unconsciousness, df_consciousness = separate_consciousness_state(dataset_dir, user_id, change_columns)
        user_dict[user_id] = {0: df_unconsciousness, 1: df_consciousness}
    return user_dict


# Maximize data-use: Select data from intervals

def select_consequitive_rows(dataframe, int_length):
    n = random.randint(0, len(dataframe)-int_length)
    return dataframe[n:n+int_length].reset_index(drop=True)

def select_consequitive_rows_intervals(dataframe, unconc_num_ints, int_length):
    data_all_intervals = []
    interval_size = int(np.floor(len(dataframe) / unconc_num_ints))

    end_index = interval_size
    while (end_index < len(dataframe)):
        data_all_intervals.append(select_consequitive_rows(dataframe[(end_index-interval_size):end_index], int_length))
        end_index += interval_size # int_length

    return data_all_intervals

def store_all_interval_data(data_per_user_dict, unconc_num_ints, int_length):
    data_per_user_dict_new = {0: [], 1: []}
    
    for key in data_per_user_dict.keys():
        df_unconsciousness = data_per_user_dict[key][0]
        df_consciousness = data_per_user_dict[key][1]
        
        data_per_user_dict_new[0].extend(select_consequitive_rows_intervals(df_unconsciousness, unconc_num_ints, int_length))
        data_per_user_dict_new[1].append(select_consequitive_rows(df_consciousness, int_length))
    return data_per_user_dict_new 