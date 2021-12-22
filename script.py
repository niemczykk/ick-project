import librosa
import numpy as np
import sys
from keras.models import load_model
import time
import serial


def extract_features():
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    stft = np.abs(librosa.stft(X))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz


file_path = sys.argv[1]

# Odczytanie modelu
model = load_model('saved_model')

#  Extrakcja cech
with open(file_path, 'r') as f:
    extract = extract_features()

features = [np.concatenate((extract[0], extract[1], extract[2], extract[3], extract[4]), axis=0)]

#  Predykcja
prediction = model.predict(np.array(features))
print(prediction)
classes = np.argmax(prediction, axis=1)

if prediction[0][classes[0]] > 0.95:
    print('Class:', classes[0])
else:
    print('Class: Unknown')

# Sending data to arduino
# arduino = serial.Serial('COM1', 9600)
# time.sleep(2)
# arduino.write(classes[0])
# arduino.close()
