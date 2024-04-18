import requests
from zipfile import ZipFile
from io import BytesIO
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import math

# Download and extract dataset
# url = 'https://github.com/ieee8023/covid-chestxray-dataset/archive/refs/heads/master.zip'
# response = requests.get(url)
# zip_file = ZipFile(BytesIO(response.content))
# zip_file.extractall('dataset')

# Load metadata and prepare image paths
base_dir = 'dataset/covid-chestxray-dataset-master'
metadata_path = os.path.join(base_dir, 'metadata.csv')
metadata = pd.read_csv(metadata_path)
covid_cases = metadata[metadata['finding'].str.contains('COVID-19')]

images_dir = os.path.join(base_dir, 'images')
image_files = [os.path.join(images_dir, file) for file in covid_cases['filename'] if os.path.exists(os.path.join(images_dir, file))]

# Split dataset
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Prepare datasets
def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [150, 150])
    return img / 255.0

def load_dataset(file_paths):
    labels = [1 if 'virus' in path else 0 for path in file_paths]  # Example labeling function
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda x, y: (process_image(x), y))
    return ds.batch(20).prefetch(tf.data.AUTOTUNE)

train_ds = load_dataset(train_files)
val_ds = load_dataset(val_files)

# Model setup
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Model training
history = model.fit(train_ds, epochs=5, validation_data=val_ds)

# Save and evaluate model
model.save('covid_classifier_model.keras')
print("Final model evaluation:")
model.evaluate(val_ds)
