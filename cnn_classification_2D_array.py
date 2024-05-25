from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Internal imports
from helpers.set_up_tensor_data import set_up_tensor_data_from_image
from helpers.vgg16_net import *
from helpers.vgg19_net import VGG19
from data_preprocessing_2D_array import X_train, X_test, y_train, y_test
from helpers.evaluation_helpers import evaluate, predict, get_labels, AUC_curve, predict_probabilities

#################################################################
#################################################################

print('hello')
print(X_test.shape)
print('bye')

# ========
# Datasets
# ========

# experiment
exp_name = 'exp2'

# Define relevant directories
base_dir = Path(os.getcwd()).parent.absolute()
print(f'current directory: {base_dir}')

dataset_dir = os.path.join(base_dir, f'datasets/image_data/{exp_name}')
print(f'datasets directory: {dataset_dir}')

batch_size = 5
nr_epochs = 3
class_names = [0, 1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

######

num_rows, num_cols = X_train[0].shape

# Train model
model = VGG19(class_names, num_rows, num_cols)
model.fit(train_dataset, validation_data=test_dataset, epochs=nr_epochs)


# Save model
models_path = os.path.join(base_dir, 'models')
model_name = f'model{exp_name}.keras'
model.save(os.path.join(models_path, model_name))

# Load model
model = keras.models.load_model(os.path.join(models_path, model_name))

# ==========
# Evaluation
# ==========

# Get accuracy and loss
# evaluate(model, test_dataset, batch_size, 'VGG-16')

# Get the relevant values for evaluation
vgg16_predicted = predict(model, test_dataset)
vgg16_labels = get_labels(test_dataset)

# AUC ROC and AUC PR scores
print('_________________________________________________________________')
AUC_ROC, AUC_ROC_result = AUC_curve('ROC', vgg16_labels, vgg16_predicted)
print("AUC ROC score:", AUC_ROC_result.numpy())
AUC_PR, AUC_PR_result = AUC_curve('PR', vgg16_labels, vgg16_predicted)
print("AUC PR score:", AUC_PR_result.numpy())
print('_________________________________________________________________')

# Confusion metric
conf_matrix = confusion_matrix(vgg16_labels, vgg16_predicted)
TN, FP, FN, TP = conf_matrix.ravel()

# Print the results
print("True Positives:", TP)
print("False Positives:", FP)
print("True Negatives:", TN)
print("False Negatives:", FN)
print('_________________________________________________________________')

# Visuals
y_test = y_test
X_test = X_test

for i in range(len(y_test)):
    X_point = X_test[i]
    y_point = y_test[i]

    #sample_dataset = tf.data.Dataset.from_tensor_slices((X_point, y_point))
    #sample_dataset = sample_dataset.batch(batch_size).prefetch(AUTOTUNE)

    # Convert to TensorFlow Dataset
    X_point_batched = np.expand_dims(X_point, axis=0)  # Batch the single data point
    y_point_batched = np.array([y_point])

    sample_dataset = tf.data.Dataset.from_tensor_slices((X_point_batched, y_point_batched))  # Include comma to create tuple
    sample_dataset = sample_dataset.batch(1)  # Batch size of 1
    sample_dataset = sample_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    sample_predicted = predict_probabilities(model, sample_dataset)
    sample_predicted_rounded = round(sample_predicted[0][1], 3)

    fig, ax = plt.subplots(nrows=20, figsize=(10, 10), sharey=True)

    # Find the overall minimum and maximum values across all line plots
    global_min = np.min(X_point)
    global_max = np.max(X_point)

    for j in range(20):
        ax[j].plot(np.arange(30), X_point[:, j], linewidth=1)
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['bottom'].set_visible(False)
        ax[j].spines['left'].set_visible(False)

        # Set the same y-axis limits for all subplots
        ax[j].set_ylim(global_min, global_max)

    plt.suptitle(f'True: {y_point}, Predicted: {sample_predicted}')
    plt.show()
