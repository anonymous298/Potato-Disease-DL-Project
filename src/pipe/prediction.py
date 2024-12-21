import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('model/model.h5')

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=(256, 256))  # Adjust size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

def predict(img_path):
    """Make a prediction using the loaded model."""
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    
    # Assuming the model outputs probabilities for 3 classes
    class_names = ['early_blight', 'healthy', 'late_blight']
    predicted_class_index = np.argmax(predictions)  # Get index of highest probability
    predicted_class_name = class_names[predicted_class_index]  # Map index to class name
    
    return predicted_class_name