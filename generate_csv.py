import pandas as pd
import numpy as np  # Add this line
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
model = load_model('cancer_detection_model.h5')

# Configure the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test/',           # Directory containing 'unknown' subfolder with images
    target_size=(96, 96),
    batch_size=32,     # Increase batch size to reduce iteration overhead
    class_mode=None,
    shuffle=False,
    color_mode='rgb'
)

print("Number of images found:", len(test_generator.filepaths))

# Generate predictions in smaller batches
try:
    predictions = []
    for batch in test_generator:
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds)
        if len(predictions) >= len(test_generator.filepaths):
            break

    # Convert predictions to binary labels
    predicted_labels = (np.array(predictions) > 0.5).astype(int).ravel()

    # Create and save submission file
    submission = pd.DataFrame({
        'id': [filename.split('/')[-1].replace('.tif', '') for filename in test_generator.filepaths],
        'label': predicted_labels
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

except Exception as e:
    print("Error during prediction:", e)
