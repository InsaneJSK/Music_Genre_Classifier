import numpy as np
import joblib
from tensorflow.keras.models import load_model
from .audio_utils import extract_features

# Load saved components
model = load_model("models/genre_classifier_model.keras")  # or 'model.h5'
pipeline = joblib.load("models/preprocessing_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def predict_genre(features: np.ndarray) -> str:
    # Ensure input is 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Transform using pipeline
    transformed = pipeline.transform(features)

    # Predict using the model
    probs = model.predict(transformed)
    predicted_index = np.argmax(probs, axis=1)[0]
    
    # Decode label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label


# For testing
if __name__ == "__main__":
    url = input("Enter file_path: ")
    sample = extract_features(url)
    prediction = predict_genre(sample)
    print("Predicted Genre:", prediction)
