from pathlib import Path

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


# Interpolation method
interp_method = 'lanczos'


# experiment
exp_dir = 'exp2'


# Save the image data
for i in range(0, 2):
    create_directory(f'{dataset_dir_write}/{exp_dir}/state{i}')
    save_spectrogram(i, sliced_data_dict, interp_method, dataset_dir_write, exp_dir)
    save_spectrogram(i, sliced_data_dict, interp_method, dataset_dir_write, exp_dir)