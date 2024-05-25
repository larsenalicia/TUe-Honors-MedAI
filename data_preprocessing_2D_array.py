from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Internal imports
from helpers.data_preprocessing_helpers import column_renaming, store_data_per_user, store_all_interval_data
from helpers.data_to_heatmap import create_directory, save_spectrogram

# =================================================


# Define relevant directories
base_dir = Path(os.getcwd()).parent.absolute()
print(f'current directory: {base_dir}')

dataset_dir_read = os.path.join(base_dir, 'datasets/source_data/PhysioNet/OR')
print(f'datasets directory: {dataset_dir_read}')

dataset_dir_write = os.path.join(base_dir, 'datasets/image_data')
print(f'datasets directory: {dataset_dir_write}')


# Combine datasets _Sdb and _l data
change_columns = column_renaming()
data_per_user_dict = store_data_per_user(dataset_dir_read, change_columns)


# Maximize data-use: Select data from intervals
int_length = 30
unconc_num_ints = 15
sliced_data_dict = store_all_interval_data(data_per_user_dict, unconc_num_ints, int_length)


# Array of train and test data
X_np_array_0 = np.array(list(map(lambda x: x.to_numpy(), sliced_data_dict[0])))
X_np_array_1 = np.array(list(map(lambda x: x.to_numpy(), sliced_data_dict[1])))
# X = np.append(X_np_array_0, X_np_array_1, axis=0)
X = np.append(X_np_array_0, np.repeat(X_np_array_1, 15, axis=0), axis=0)

y_np_array_0 = np.full(len(X_np_array_0), 0)
y_np_array_1 = np.full(len(X_np_array_1), 1)
# y = np.append(y_np_array_0, y_np_array_1, axis=0)
y = np.append(y_np_array_0, np.repeat(y_np_array_1, 15, axis=0), axis=0)

# More proper re-sampling (Not showing seen data in test data)
X_np_array_0 = np.array(list(map(lambda x: x.to_numpy(), sliced_data_dict[0])))
X_np_array_1 = np.array(list(map(lambda x: x.to_numpy(), sliced_data_dict[1])))
y_np_array_0 = np.full(len(X_np_array_0), 0)
y_np_array_1 = np.full(len(X_np_array_1), 1)

X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_np_array_0, y_np_array_0, test_size=0.33, random_state=42, shuffle=True)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_np_array_1, y_np_array_1, test_size=0.33, random_state=42, shuffle=True)

X_train = np.append(X_train_0, np.repeat(X_train_1, 15, axis=0), axis=0)
X_test = np.append(X_test_0, np.repeat(X_test_1, 15, axis=0), axis=0)
y_train = np.append(y_train_0, np.repeat(y_train_1, 15, axis=0), axis=0)
y_test = np.append(y_test_0, np.repeat(y_test_1, 15, axis=0), axis=0)

print(len(y_np_array_0), len(y_np_array_1), len(np.repeat(y_np_array_1, 15, axis=0)))
print(X.shape, y.shape)
