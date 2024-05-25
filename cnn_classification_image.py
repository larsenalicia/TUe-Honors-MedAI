from tensorflow import keras
from pathlib import Path

# Internal imports
from helpers.set_up_tensor_data import set_up_tensor_data_from_image
from helpers.vgg16_net import *
from helpers.evaluation_helpers import *

#################################################################
#################################################################

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


# Open an (arbitrary) image file and check it's size
image = Image.open(os.path.join(f'{dataset_dir}/state0', os.listdir(f'{dataset_dir}/state0')[0]))
img_width, img_height = image.size

# Set up tensor-data
batch_size = 5
train_ds, val_ds, class_names = set_up_tensor_data_from_image(dataset_dir, img_height, img_width, batch_size)

# =========================
# Model set-up and training
# =========================

# Train model
vgg16_model = VGG16model(class_names, img_height, img_width)
vgg16_model.fit(train_ds, validation_data=val_ds, epochs=1)

# Save model
models_path = os.path.join(base_dir, 'models')
model_name = f'vgg16_model{exp_name}.keras'
vgg16_model.save(os.path.join(models_path, model_name))

# Load model
vgg16_model = keras.models.load_model(os.path.join(models_path, model_name))

# ==========
# Evaluation
# ==========

# Get accuracy and loss
evaluate(vgg16_model, val_ds, batch_size, 'VGG-16')

# Get the relevant values for evaluation
vgg16_predicted = predict(vgg16_model, val_ds)
vgg16_labels = get_labels(val_ds)

# AUC ROC and AUC PR scores
AUC_ROC, AUC_ROC_result = AUC_curve('ROC', vgg16_labels, vgg16_predicted)
print("AUC ROC score:", AUC_ROC_result.numpy())
AUC_PR, AUC_PR_result = AUC_curve('PR', vgg16_labels, vgg16_predicted)
print("AUC PR score:", AUC_PR_result.numpy())
