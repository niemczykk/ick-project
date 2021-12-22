import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder, StandardScaler


def extract_features(files):
    file_name = files.file

    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    stft = np.abs(librosa.stft(X))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz, files.label


# Zebranie danych
kristina = os.listdir('Kristina_TEST')
kri_df = pd.DataFrame(kristina)
kri_df.loc[0:25, ['label', 'speaker']] = ['1', '3']
kri_df = kri_df.rename(columns={0: 'file'})
for i in range(0, 25):
    kri_df.at[i, 'file'] = 'Kristina_TEST/' + kri_df.at[i, 'file']

lukasz = os.listdir('Lukasz_TEST')
luk_df = pd.DataFrame(lukasz)
luk_df.loc[0:25, ['label', 'speaker']] = ['0', '0']
luk_df = luk_df.rename(columns={0: 'file'})
for i in range(0, 25):
    luk_df.at[i, 'file'] = 'Lukasz_TEST/' + luk_df.at[i, 'file']

krzysztof = os.listdir('Krzysztof_TEST')
krz_df = pd.DataFrame(krzysztof)
krz_df.loc[0:25, ['label', 'speaker']] = ['0', '2']
krz_df = krz_df.rename(columns={0: 'file'})
for i in range(0, 25):
    krz_df.at[i, 'file'] = 'Krzysztof_TEST/' + krz_df.at[i, 'file']

szymon = os.listdir('Szymon_TEST')
szy_df = pd.DataFrame(szymon)
szy_df.loc[0:25, ['label', 'speaker']] = ['0', '1']
szy_df = szy_df.rename(columns={0: 'file'})
for i in range(0, 25):
    szy_df.at[i, 'file'] = 'Szymon_TEST/' + szy_df.at[i, 'file']

frames = [kri_df, krz_df, luk_df, szy_df]
df = pd.concat(frames)

# Zaszumianie zbioru
df = df.sample(frac=1).reset_index(drop=True)

# Dzielenie na zbiory uczace
df_train = df[:69]
df_val = df[70:89]
df_test = df[90:99]

# Ekstrakcja cech
features_label = df.apply(extract_features, axis=1)

# Polaczenie cech w jedna tablice
features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0],
                                    features_label[i][1],
                                    features_label[i][2],
                                    features_label[i][3],
                                    features_label[i][4]),
                                   axis=0))

X = np.array(features)
y = np.asarray(df['speaker'])

lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))

# Dzielenie zbiorow na treningowy, testowy i walidacyjny
X_train = X[:69]
y_train = y[:69]

X_val = X[70:89]
y_val = y[70:89]

X_test = X[90:99]
y_test = y[90:99]

# Dopasowywanie danych
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

# Utworzenie modelu
model = Sequential()

# Dodawanie warstw
model.add(Dense(193, input_shape=(193,), activation='relu', name='First'))
model.add(Dropout(0.1))

model.add(Dense(128, activation='relu', name='Second'))
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu', name='Third'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax', name='Last'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', run_eagerly=True)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

history = model.fit(X_train, y_train, batch_size=256, epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

# Zapisanie modelu do pozniejszego uzytku
model.save('saved_model')

score = model.evaluate(X_test, y_test, verbose=0)
print('Loss', score[0])
print('Accu', score[1])

# Utworzenie grafu
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.figure(figsize=(12, 8))
plt.plot(train_accuracy, label='Skuteczność treningu', color='blue')
plt.plot(val_accuracy, label='Skuteczność walidacji', color='orange')

plt.title('Skuteczność walidacji i treningu na podstawie epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Categorical Crossentropy', fontsize=18)
plt.xticks(range(0, 100, 5), range(0, 100, 5))

plt.legend(fontsize=18)
plt.show()
