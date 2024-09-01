# Pneumonia-Detection

### Dataset Overview:
The "Pneumonia Detection Using Chest X-Ray Images" project aims to develop a robust and accurate model for the detection of pneumonia from chest X-ray images in pediatric patients aged one to five years old. The dataset comprises 5,863 X-Ray images categorized into Pneumonia and Normal classes. These images were sourced from the Guangzhou Women and Children’s Medical Center as part of routine clinical care. The dataset is split into three folders: train, test, and val.

Dataset Source:
https://data.mendeley.com/datasets/rscbjbr9sj/2

### Dataset Description:
* Categories: Pneumonia and Normal
* Image Types: JPEG format
* Imaging Method: Anterior-posterior chest X-rays
* Quality Control: All images underwent initial quality screening to remove poor-quality scans. Diagnoses were assessed by two expert physicians before being used for AI training.
* Evaluation: A third expert reviewed the evaluation set to ensure grading accuracy.

### Dataset Folder Structure:
* Train Directory: Contains training images for model training.
* Test Directory: Contains images for model testing.
* Val Directory: Includes images for validation during model training.

## Overview

This repository contains a Python implementation of a deep learning model for pneumonia diagnosis using TensorFlow and Keras. The model leverages both custom and pre-trained architectures to improve diagnostic accuracy.

## Key Components

### Initial Model

- **Architecture**: Custom neural network
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Class Weights**: Calculated to address class imbalance

### Transfer Learning Model (InceptionV3)

- **Base Model**: InceptionV3 pre-trained on ImageNet
- **Custom Layers**: Added for fine-tuning
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: RMSprop

## Prerequisites

Ensure you have the following Python packages installed:

- `os`
- `matplotlib` (including `matplotlib.image` and `matplotlib.pyplot`)
- `seaborn`
- `pandas`
- `numpy`
- `tensorflow` (including `keras`, `tensorflow.keras.preprocessing.image`, `tensorflow.keras.optimizers`, `tensorflow.keras.utils`, `tensorflow.keras.applications.inception_v3`, `tensorflow.keras.layers`)
- `kaggle` (for dataset download and management)

You can install the required packages using pip:

```bash
pip install matplotlib seaborn pandas numpy tensorflow kaggle
```

## Setup

### Kaggle API Configuration

Ensure you have your Kaggle API key (`kaggle.json`). Set up the Kaggle API by running the following commands:

```bash
! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
```

## Download and Unzip the Dataset

Download the dataset from Kaggle and unzip it:

```bash
! kaggle datasets download paultimothymooney/chest-xray-pneumonia
! unzip /content/chest-xray-pneumonia.zip
```
## Data Preparation

Place your dataset into the appropriate directories as follows:

- `train_NORMAL_dir`: Directory containing normal X-ray images for training.
- `train_PNEUMONIA_dir`: Directory containing pneumonia X-ray images for training.
- `val_NORMAL_dir`: Directory containing normal X-ray images for validation.
- `val_PNEUMONIA_dir`: Directory containing pneumonia X-ray images for validation.


# Deep Learning Model Training and Evaluation

This repository contains code for training and evaluating a deep learning model using TensorFlow and Keras. It includes implementations for calculating recall, precision, and F1 score metrics, training a model with custom metrics, and leveraging transfer learning with the InceptionV3 architecture.

## Custom Metrics

### Recall

```python
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
```

### Precision
```python
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
```

### F1 Score
```python
def f1(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1 = recall(y_true, y_pred)
    return 2 * ((precision1 * recall1) / (precision1 + recall1 + K.epsilon()))
```

## Model Training

### Initial Model

- **Optimizer**: RMSprop
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, F1 Score, Precision, Recall
- **Class Weights**: Adjusted to handle class imbalance

```python
weight_0 = (len(os.listdir(train_NORMAL_dir)) + len(os.listdir(train_NORMAL_dir))) / (2 * len(os.listdir(train_NORMAL_dir)))
weight_1 = (len(os.listdir(train_NORMAL_dir)) + len(os.listdir(train_NORMAL_dir))) / (2 * len(os.listdir(train_NORMAL_dir)))
class_weights = {0: weight_0, 1: weight_1}

history = model.fit(train_generator,
                    steps_per_epoch=40,
                    epochs=30,
                    validation_data=val_generator,
                    validation_steps=2,
                    class_weight=class_weights,
                    verbose=1)
```

## Transfer Learning with InceptionV3

- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Input Shape**: (128, 128, 3)
- **Custom Input Layer**: Converts grayscale images to RGB
- **Custom Layers**: Flatten, Dense (512 units), Dropout, Dense (1 unit)

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

pre_trained_model = InceptionV3(input_shape=(128, 128, 3), include_top=False)

for layer in pre_trained_model.layers:
    layer.trainable = False

model = tf.keras.models.Model(inputs=pre_trained_model.inputs, outputs=pre_trained_model.get_layer('mixed7').output)

input_tensor = Input(shape=(128, 128, 1))
x = Conv2D(3, (3, 3), padding='same')(input_tensor)
x = model(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=input_tensor, outputs=out)
```

## Training Parameters for Transfer Learning Model

- **Optimizer**: RMSprop with learning rate 0.0001
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, F1 Score, Precision, Recall

```python
history = model.fit(train_generator,
                    steps_per_epoch=40,
                    epochs=15,
                    validation_data=val_generator,
                    validation_steps=2,
                    verbose=1)
```


## Results

### Initial Model

- **Loss**: 0.3317
- **Accuracy**: 0.8974
- **F1 Score**: 0.9212
- **Precision**: 0.8684
- **Recall**: 0.9814

### Transfer Learning Model

- **Loss**: 0.3956
- **Accuracy**: 0.8622
- **F1 Score**: 0.8988
- **Precision**: 0.8280
- **Recall**: 0.9845

## Observations

- **Initial Model**: Recall increased when considering class weights, but accuracy and precision decreased.
- **Transfer Learning Model**: Higher recall, precision, and accuracy compared to the initial model.

## Plots

Training and validation accuracy and loss are plotted per epoch for both models. Use the following commands to generate and view the plots:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], 'bo', color='#ff0066')
plt.plot(history.history['val_accuracy'], color='#00ccff')
plt.title('Training and validation accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.figure()

plt.plot(history.history['loss'], 'bo', color='#ff0066')
plt.plot(history.history['val_loss'], color='#00ccff')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Training and validation loss')
```

## Summary
This repository demonstrates the use of custom metrics and transfer learning to improve model performance for binary classification tasks. The InceptionV3 model provides a robust feature extraction backbone, while custom metrics help evaluate model performance effectively.






