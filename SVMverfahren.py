import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Feature extraction function with minimal features
def extract_features(audio, sr, n_mfcc=20, n_fft=256, hop_length=128, max_pad_len=200):
    if len(audio) < n_fft:
        n_fft = len(audio)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs.flatten()

# Augmentation: Add noise
def augment_audio(audio):
    noise = np.random.randn(len(audio)) * 0.005
    return audio + noise

# Process a single file
def process_file(file_path, label):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # Lower sample rate for speed
        features = [extract_features(audio, sr)]  # Original
        augmented_audio = augment_audio(audio)
        features.append(extract_features(augmented_audio, sr))  # Augmented
        labels = [label] * len(features)
        return features, labels
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return [], []

# Load dataset
def load_data(audio_folder_path):
    files = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith(".wav")]
    results = Parallel(n_jobs=-1)(delayed(process_file)(
        file, 0 if "felix" in os.path.basename(file).lower() else 1 if "linelle" in os.path.basename(file).lower() else 2
    ) for file in files)
    
    features, labels = [], []
    for f, l in results:
        features.extend(f)
        labels.extend(l)
    
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Aucune caractéristique ou étiquette valide n'a été trouvée dans le dossier.")
    
    return np.array(features), np.array(labels)

# Train and evaluate model
def train_svm_model(path):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight='balanced')  # Pre-defined parameters
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")  # Accuracy in percentage
    y_pred = model.predict(X_test)

    # Classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Felix", "Linelle", "Unknown"], yticklabels=["Felix", "Linelle", "Unknown"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return model, scaler

# Predict speaker
def predict_speaker(model, audio_file, scaler):
    try:
        audio, sr = librosa.load(audio_file, sr=16000)  # Lower sample rate for speed
        features = extract_features(audio, sr)
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        speaker = ["Felix", "Linelle", "Unknown"][prediction]
        print(f"File: {audio_file}, Predicted Speaker: {speaker}")
        return speaker
    except Exception as e:
        print(f"Erreur lors de la prédiction du fichier {audio_file}: {e}")
        return "Erreur"

# Main execution
if __name__ == "__main__":
    audio_folder = r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen"
    model, scaler = train_svm_model(audio_folder)

    test_files = [
        r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen\Felix_1_1.wav",
        r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen\Felix_15_2.wav",
        r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen\Linelle_7_1.wav",
        r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen\Linelle_10_2.wav",
        r"C:\Users\Yvann\Desktop\New folder\MySVM\MySVM\Stimmen\Unknown_1.wav"
    ]

    for file in test_files:
        predict_speaker(model, file, scaler)
