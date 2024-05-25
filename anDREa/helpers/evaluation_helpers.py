import numpy as np
import tensorflow as tf


def evaluate(model, dataset, batch_size, name='model'):
    loss, acc = model.evaluate(dataset, batch_size=batch_size)
    print(f"{name}'s accuracy: {round((acc * 100), 2)}%")


def evaluate_xy(model, x, y, batch_size, name='model'):
    loss, acc = model.evaluate(x, y, batch_size=batch_size)
    print(f"{name}'s accuracy: {round((acc * 100), 2)}%")


def predict(model, dataset):
    return np.argmax(model.predict(dataset), axis=-1)


def predict_probabilities(model, dataset):
    return model.predict(dataset)


def predict_xy(model, x, y):
    return np.argmax(model.predict(x, y), axis=-1)


def get_labels(dataset):
    all_val_labels = []
    for _, labels in dataset:
        all_val_labels.extend(labels.numpy())
    all_labels = np.array(all_val_labels)
    return all_labels


def get_labels_xy(x, y):
    all_val_labels = []
    for _, labels in x, y:
        all_val_labels.extend(labels.numpy())
    all_labels = np.array(all_val_labels)
    return all_labels


def AUC_curve(curve_type: str, vgg16_labels, vgg16_predict):
    AUC_CURVE = tf.keras.metrics.AUC(
        num_thresholds=200,
        curve=curve_type,
        summation_method='interpolation',
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False
    )
    AUC_CURVE.update_state(vgg16_labels, vgg16_predict)
    AUC_ROC_result = AUC_CURVE.result()
    return AUC_CURVE, AUC_ROC_result
