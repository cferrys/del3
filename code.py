import os
import pandas as pd
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_DIR = '.'
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001

def load_data(data_dir):
    """Load and preprocess dataset."""
    try:
        labels_df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
        labels_df['label'] = labels_df['label'].astype(str)
        labels_df['id'] = labels_df['id'] + '.tif'
        logging.info("Data loaded successfully.")
        return labels_df
    except FileNotFoundError as e:
        logging.error("File not found. Ensure 'train_labels.csv' exists in the specified directory.")
        raise e

def visualize_data(df):
    """Visualize data distribution and sample images."""
    sns.countplot(x='label', data=df)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

    # Show sample images
    sample_images = df.sample(4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, row in enumerate(sample_images.itertuples()):
        img_path = os.path.join(DATA_DIR, 'train', row.id)
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {row.label}")
        axes[i].axis('off')
    plt.show()

def get_data_generators(df, data_dir, img_size, batch_size):
    """Prepare data generators with augmentations."""
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_generator, val_generator

def build_model(input_shape):
    """Build and compile CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    logging.info("Model built and compiled successfully.")
    return model

def train_and_evaluate_model(model, train_generator, val_generator, epochs):
    """Train the model and evaluate its performance."""
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def main():
    labels_df = load_data(DATA_DIR)
    visualize_data(labels_df)
    train_gen, val_gen = get_data_generators(labels_df, DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_model((*IMG_SIZE, 3))
    train_and_evaluate_model(model, train_gen, val_gen, DEFAULT_EPOCHS)
    model.save('cancer_detection_model.h5')
    logging.info("Model saved as 'cancer_detection_model.h5'.")

if __name__ == "__main__":
    main()
