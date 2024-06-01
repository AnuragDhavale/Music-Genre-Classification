# pip install "librosa==0.9.1" for this file

import os
import h5py
import librosa
import itertools
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.model_selection import train_test_split

# os.chdir(r'/content/drive/My Drive/')

np.random.seed(42)

def splitsongs(X, y, window = 0.05, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

#generating np array of mel spectrograms from list of songs

def gen_melspectrograms(songs, sample_rate=660000, n_fft=1024, hop_length=256):
    mels = lambda x: librosa.feature.melspectrogram(y=x, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)[:,:,np.newaxis]
    tsongs = map(mels, songs)
    return np.array(list(tsongs))

def split_convert(x, y):
  array_spec, array_genre = [], []
    # Convert to spectrograms and split into small windows
  for fn, genre in tqdm(zip(x, y),total=len(y),desc='Processing Audio Files'):
    print(fn, end="\n")
    try:
      signal, sr = sf.read(fn)
    except:
      print(fn, "<---- fix this file\n")
    signal = signal[:660000]

    # Convert to dataset of spectograms/melspectograms
    signals, y = splitsongs(signal, genre, window=0.05) #keep window=0.05. Other values are for experimenting.

    # Convert to "spec" representation
    specs = gen_melspectrograms(signals)

    # Save files
    array_genre.extend(y)
    array_spec.extend(specs)

  return np.array(array_spec), to_categorical(array_genre)

def read_data(src_dir, genres, song_samples):
    # Empty array of dicts with the processed features from all files
    array_fn = []
    array_genres = []

    # Get file list from the folders
    for x,_ in genres.items():
        folder = os.path.join(src_dir,x)
        for root, subdirs, files in os.walk(folder):
            for file in files:
                file_name = os.path.join(root, file)

                # Save the file name and the genre
                array_fn.append(file_name)
                array_genres.append(genres[x])

    # Split into small segments and convert to spectrogram
    X_data, y_data = split_convert(array_fn, array_genres)
    return X_data, y_data

# Parameters
gtzan_dir = r'E:\Projects\MusicGenreClassification\archive\Data\genres'
song_samples = 660000
genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

# Read the data
X_data, y_data = read_data(gtzan_dir, genres, song_samples)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, stratify=y_data, random_state=42)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)