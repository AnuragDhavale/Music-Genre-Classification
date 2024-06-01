# Music Genre Classification

Music Genre Classification is a project that uses a machine learning model for a real-time prediction of the genre of an audio file being played. The model is trained on a dataset of various music genres and can provide predictions for new audio.

## Overview

The project consists of the following components:

1. **Data Preparation:** The dataset used for training and testing the model is the GTZAN dataset, which includes audio samples from various music genres. These audios are processed into NumPy arrays, which are loaded for training the machine learning model.

2. **Model Architecture:** The classification model is a Convolutional Neural Network (CNN) designed for processing mel spectrograms extracted from audio files.

3. **GUI Application:** A GUI application is provided for users to interact with the model. Users can play any audio, and the application will predict the genre using the trained model.





