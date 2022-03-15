import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import noisereduce as nr


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        X = X.flatten()
        print(X.shape)
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'angry']


def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("C:\\Users\\meett\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    print(y)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# x_train, x_test, y_train, y_test = load_data(test_size=0.25)
#
# print((x_train.shape[0], x_test.shape[0]))
#
# print(f'Features extracted: {x_train.shape[1]}')
#
# model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
#                       max_iter=500)
#
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
#
# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
#
# maxs=0
#
# print("Accuracy: {:.2f}%".format(accuracy * 100))
#
# if accuracy*100 > maxs:
#     maxs=accuracy*100
#     print(maxs)
#     with open('ser.pkl', 'wb') as file:
#         # A new file will be created
#         pickle.dump(model, file)

with open('ser.pkl', 'rb') as file:
    # Call load method to deserialze
    model = pickle.load(file)


fs = 16000
seconds = 3
print("Started:")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write('output.wav', fs, myrecording)
print("Ended")
# rate, data = wavfile.read("output.wav")
# reduced = nr.reduce_noise(y=data, sr=rate)
# wavfile.write("output_red.wav", rate=rate)
for file in glob.glob("C:\\Users\\meett\\PycharmProjects\\SER\\output.wav"):
    x = extract_feature(file, mfcc=True, chroma=True, mel=True)
y_pred = model.predict(np.array([x, ]))
print(y_pred)
# print("yes")
# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)


# print("Accuracy: {:.2f}%".format(accuracy * 100))
