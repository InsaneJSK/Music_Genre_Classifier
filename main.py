import streamlit as st
from utils.classifier import predict_genre
from utils.audio_utils import extract_features, extract_audio_from_youtube
import os

st.title("🎶 Music Genre Classifier")

# Choose input method
input_mode = st.radio("Choose input method:", ["Upload Audio File", "YouTube URL"])

if input_mode == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    st.audio(uploaded_file, format='audio/wav')
    if uploaded_file is not None:
        with st.spinner("Downloading video"):
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.read())
        with st.spinner("Processing audio"):
            features = extract_features("temp_audio.wav")
            genre = predict_genre(features)
        st.success(f"🎧 Predicted Genre: **{genre}**")
        os.remove("temp_audio.wav")

elif input_mode == "YouTube URL":
    url = st.text_input("Enter YouTube URL")
    if st.button("Process"):
        if url:
            try:
                st.video(url)
                with st.spinner("Downloading video"):
                    audio_path = extract_audio_from_youtube(url)  # <- returns path to downloaded .wav file
                with st.spinner("Processing audio"):
                    features = extract_features(audio_path)
                    genre = predict_genre(features)
                st.success(f"🎧 Predicted Genre: **{genre}**")
                os.remove(audio_path)
            except Exception as e:
                st.error(f"Failed to process: {str(e)}")
