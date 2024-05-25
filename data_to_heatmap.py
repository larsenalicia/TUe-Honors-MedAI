import numpy as np
import matplotlib.pyplot as plt
import os


def heapmap_interpolated(df_data, interp_method, impath='', save=False):
    np.random.seed(123)

    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 3),
                           subplot_kw={'xticks': [], 'yticks': []})

    ax.imshow(df_data.T, interpolation=interp_method, cmap='viridis');
    ax.set_aspect(df_data.shape[0] / df_data.shape[1])
    plt.tight_layout()

    if save:
        plt.axis('off')
        plt.savefig(impath, dpi=100, pad_inches=0.0, transparent=True, bbox_inches='tight')
    else:
        plt.show()


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists! Data may be overwritten!")


def save_spectrogram(state, sliced_data_dict, interpolation_method, dataset_dir, exp_dir):
    for i, data in enumerate(sliced_data_dict[state]):
        if state == 0:
            heapmap_interpolated(df_data=data, interp_method=interpolation_method,
                                 impath=f'{dataset_dir}/{exp_dir}/state{state}/img{i}.png', save=True)
        elif state == 1:
            for j in range(0, 15):
                heapmap_interpolated(df_data=data, interp_method=interpolation_method,
                                     impath=f'{dataset_dir}/{exp_dir}/state{state}/img{i}_{j}.png', save=True)
