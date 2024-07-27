import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Execution has started")

# Define input shape and batch size
input_shape = (150, 150, 3)
batch_size = 16  # Adjust based on GPU memory

# Directories for train, test, and validation sets
train_directory = '/content/Train/Train'
test_directory = '/content/Test/Test'
val_directory = '/content/Validation/Validation'

# Function to parse image pairs and labels
def parse_function(pair_folder):
    image1_path = tf.strings.join([pair_folder, 'image1.jpg'], separator='/')
    image2_path = tf.strings.join([pair_folder, 'image2.jpg'], separator='/')
    label_path = tf.strings.join([pair_folder, 'label.txt'], separator='/')

    image1 = tf.io.read_file(image1_path)
    image1 = tf.image.decode_jpeg(image1, channels=3)
    image1 = tf.image.resize(image1, [150, 150])
    image1 = tf.image.convert_image_dtype(image1, tf.float32) / 255.0

    image2 = tf.io.read_file(image2_path)
    image2 = tf.image.decode_jpeg(image2, channels=3)
    image2 = tf.image.resize(image2, [150, 150])
    image2 = tf.image.convert_image_dtype(image2, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.strings.to_number(label, out_type=tf.int32)

    return (image1, image2), label

# Function to create dataset from directory
def create_dataset(directory, batch_size):
    pair_folders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    dataset = tf.data.Dataset.from_tensor_slices(pair_folders)
    dataset = dataset.map(lambda x: parse_function(x), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_dataset(train_directory, batch_size)
val_dataset = create_dataset(val_directory, batch_size)
test_dataset = create_dataset(test_directory, batch_size)

# Contrastive Loss function
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Custom accuracy metric for contrastive loss
def compute_accuracy(y_true, y_pred, threshold=0.5):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < threshold, y_true.dtype)), y_true.dtype))

# Load and compile the model
model_path = '/content/drive/My Drive/TrainedSiameseModel/siamese_resnet34_contrastive_model.h5'
siamese_model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss, 'compute_accuracy': compute_accuracy})
siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss=contrastive_loss, metrics=[compute_accuracy])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

# Train the model with EarlyStopping
history = siamese_model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=20,
                            callbacks=[early_stopping])

# Save the trained model in TensorFlow SavedModel format
siamese_model.save('/content/drive/My Drive/TrainedSiameseModels/Resnet34_Extracted_faces_Patience5_min_val_loss_BSize16_FullyTrained_Siamese_Model', save_format='tf')
print("Trained Siamese model saved successfully.")

# Evaluate the model on test data
test_loss, test_accuracy = siamese_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['compute_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_compute_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
