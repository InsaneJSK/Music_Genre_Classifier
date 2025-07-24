import os
import pandas as pd
from audio_utils import extract_features
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "data/genres_original/"
GENRES = os.listdir(DATA_DIR)
FEATURES = []
LABELS = []

def process_file(file_path, genre):
    try:
        features = extract_features(file_path)
        return (features, genre)
    except Exception as e:
        print(f"Failed on {file_path}: {e}")
        return None


if __name__ == "__main__":
    print("Extracting features...")

    file_genre_pairs = []
    for genre in GENRES:
        genre_path = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        for filename in os.listdir(genre_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_path, filename)
                file_genre_pairs.append((file_path, genre))

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda args: process_file(*args), file_genre_pairs), total=len(file_genre_pairs)))

    results = [r for r in results if r is not None]
    FEATURES, LABELS = zip(*results)

    # Convert to DataFrame
    df = pd.DataFrame(FEATURES)
    df['label'] = LABELS

    # Save
    os.makedirs("features", exist_ok=True)
    df.to_csv("features/features-richer.csv", index=False)
    print("Feature extraction complete. Saved to features/features.csv")
