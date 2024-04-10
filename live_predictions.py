import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import keras
import pathlib
import os

working_dir_path = pathlib.Path().absolute()
# Set the paths and file names
MODEL_DIR_PATH = os.path.join(str(working_dir_path), 'model')
MODEL_PATH = os.path.join(MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model.h5')

class LivePredictions:
    def __init__(self, model_path):
        self.loaded_model = keras.models.load_model(model_path)

    def make_predictions(self, audio_file):
        data, sampling_rate = librosa.load(audio_file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions_class = self.loaded_model.predict(x)
        predictions = predictions_class.argmax(axis=-1)
        return self.convert_class_to_emotion(predictions)

    @staticmethod
    def convert_class_to_emotion(pred):
        label_conversion = {'0': 'neutral', '1': 'calm', '2': 'happy', '3': 'sad', '4': 'angry', '5': 'fearful', '6': 'disgust', '7': 'surprised'}
        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
                return label

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an audio file", filetypes=[("Audio files", "*.wav *.mp3")])

    if file_path:
        # Create an instance of LivePredictions
        live_prediction = LivePredictions(model_path=MODEL_PATH)

        # Perform emotion prediction on the selected audio file
        emotion = live_prediction.make_predictions(file_path)
        print(live_prediction.loaded_model.summary())
        print(f"Emotion prediction for the selected file is: {emotion}")


if __name__ == '__main__':
    open_file_dialog()
    
