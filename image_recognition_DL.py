import random
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Input

def main():
    # Path to the image dataset dicionery
    img_dir = 'img'

    # Img dimensions
    img_height = 180
    img_width = 180

    # Model training configuration
    batch_size = 40
    num_classes = 4 # Number of classes/subfolders in img_dir
    epochs = 18

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        img_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=42,
        subset="training",
        validation_split=0.2,
    )

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        img_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=42,
        subset="validation",
        validation_split=0.2,
    )

    # Dispayin class names
    class_names = train_ds.class_names
    print(class_names)

    # Normalizing pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Defining model
    model = tf.keras.Sequential([
        Input(shape=(img_height, img_width, 3)),                # Input layer (RGB image)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 1st convolution layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),         # 1st pooling layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 2st convolution layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),         # 2st pooling layer
        tf.keras.layers.Flatten(),                              # Flatten to 1D before dense layers
        tf.keras.layers.Dense(128, activation='relu'),          # Fully connected hidden layer

        tf.keras.layers.Dense(num_classes, activation='softmax')# Output layer with softmax
    ])

    # Compile the model with appropriate loss function and optimizer
    model.compile(
        loss='sparse_categorical_crossentropy',     # For integer-labeled classes
        optimizer='adam',                           # Adaptive optimizer
        metrics=['accuracy']                        # Monitor accuracy during training
    )

    # Training model and data validation
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Starting visualisation
    plt.figure(figsize=(12, 12))
    samples = []

    # Random sample up to 4 images from each class for prediction visualization
    for c_name in class_names:
        folder = os.path.join(img_dir, c_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        chosen = random.sample(files, min(4, len(files)))
        for f in chosen:
            samples.append((os.path.join(folder, f), c_name))

    # Predict each sample image and display it with predicted label and confidence
    for i, (img_path, true_label) in enumerate(samples):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0 # Normalaize to match training input
        predc = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        pred_label = class_names[np.argmax(predc)]
        confidence = np.max(predc)

        # Show image and prediction results
        plt.subplot(4,4, i+1)
        plt.imshow(img)
        plt.title(f'Real: {true_label}\nPredicted: {pred_label} {confidence:.2f}')
        plt.axis('off')

    # Displaying all predictions images
    plt.show()

if __name__ == '__main__':
    main()