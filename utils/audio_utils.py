import librosa
import numpy as np
import pandas as pd
import yt_dlp
from pydub import AudioSegment
import os
import uuid

def extract_features(file_path, duration=30):
    y, sr = librosa.load(file_path, duration=duration)
    
    # Base features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Additional features
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # Combine all features (mean + std)
    features = np.hstack([
        np.mean(mfccs, axis=1),     np.std(mfccs, axis=1),
        np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
        np.mean(chroma, axis=1),   np.std(chroma, axis=1),
        np.mean(zcr),              np.std(zcr),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(rolloff),          np.std(rolloff),
        np.mean(tonnetz, axis=1),  np.std(tonnetz, axis=1),
        tempo,                     # scalar
        np.mean(rms),              np.std(rms)
    ])

    return  pd.DataFrame([features])

def extract_audio_from_youtube(url: str, output_format: str = "wav") -> str:
    """
    Extracts audio from a YouTube URL and returns path to the audio file (.wav).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    temp_id = str(uuid.uuid4())
    output_dir = os.path.join(temp_dir, temp_id)
    output_mp3 = f"{output_dir}.mp3"
    output_wav = f"{output_dir}.{output_format}"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_dir,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Convert to wav
        audio = AudioSegment.from_mp3(output_mp3)
        audio.export(output_wav, format="wav")

        os.remove(output_mp3)
        return output_wav

    except Exception as e:
        if os.path.exists(output_mp3): os.remove(output_mp3)
        if os.path.exists(output_wav): os.remove(output_wav)
        raise RuntimeError(f"Error extracting audio: {e}")

if __name__ == "__main__":
    url = input("Enter yt url: ")
    file_path = extract_audio_from_youtube(url=url)
    features = extract_features(file_path=file_path)
    print(features)
