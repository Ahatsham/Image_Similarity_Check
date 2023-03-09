
# -*- coding: utf-8 -*-

from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="",
    project_name="",
    workspace="",
)

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import PIL
import scipy.integrate as integrate
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

img_size = (224, 224)

os.chdir('/work/netthinker/shared/flowers')

# Create an ImageDataGenerator for the training set
train_Original_datagen = ImageDataGenerator(
    rescale=1./255,
)
train_Original_generator = train_Original_datagen.flow_from_directory(
    'train/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)

# Create an ImageDataGenerator for the validation set
valid_Original_datagen = ImageDataGenerator(rescale=1./255)
valid_Original_generator = valid_Original_datagen.flow_from_directory(
    'validation/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)

X_Original_train, y_Original_train = next(train_Original_generator)
X_Original_test, y_Original_test = next(valid_Original_generator)

# For Cezanne-People Dataset
#os.chdir('/work/netthinker/shared/Converted_Images_Ahatsham/Converted-Images-flowers-CycleGAN-Cezanne-People-Ahatsham/')
os.chdir('/work/netthinker/shared/flowers-drawing/')

train_Cezanne_datagen = ImageDataGenerator(
    rescale=1./255,
)
train_Cezanne_generator = train_Cezanne_datagen.flow_from_directory(
    'train/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)

# Create an ImageDataGenerator for the validation set
valid_Cezanne_datagen = ImageDataGenerator(rescale=1./255)
valid_Cezanne_generator = valid_Cezanne_datagen.flow_from_directory(
    'validation/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)

X_Cezanne_train, y_Cezanne_train = next(train_Cezanne_generator)
X_Cezanne_test, y_Cezanne_test = next(valid_Cezanne_generator)

#Converting Lablels into scalar vectors
y_Original_train = np.argmax(y_Original_train, axis=1)
y_Cezanne_train = np.argmax(y_Cezanne_train, axis=1)
y_Original_test = np.argmax(y_Original_test, axis=1)
y_Cezanne_test = np.argmax(y_Cezanne_test, axis=1)

def make_pairs(x_o, y_o, x_c, y_c):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = 5
    digit_indices = [np.where(y_c == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x_o)):
        # add a matching example
        x1 = x_o[idx1]
        label1 = y_o[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x_c[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x_c[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


# make train pairs
pairs_train, labels_train = make_pairs(X_Original_train, y_Original_train,X_Cezanne_train, y_Cezanne_train)
pairs_test, labels_test = make_pairs(X_Original_test, y_Original_test, X_Cezanne_test, y_Cezanne_test)

x_train_1 = pairs_train[:, 0]  
x_train_2 = pairs_train[:, 1]
x_test_1 = pairs_test[:, 0]  
x_test_2 = pairs_test[:, 1]

'''
def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 224, 224).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()
'''

#visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=5)
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

#CNN

input = tf.keras.layers.Input((224, 224, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = tf.keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(10, activation="tanh")(x)
embedding_network = tf.keras.Model(input, x)


input_1 = tf.keras.layers.Input((224, 224, 3))
input_2 = tf.keras.layers.Input((224, 224, 3))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

epochs = 200
batch_size = 32
margin = 1

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_test_1, x_test_2], labels_test),
    batch_size=batch_size,
    epochs=epochs,
)

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)