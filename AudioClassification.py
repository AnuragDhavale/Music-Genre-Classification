import tkinter as tk
import librosa
import pyaudio
import numpy as np
from tkinter import TclError
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import struct
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import GlorotUniform, Zeros

# Initialize the root window
root = tk.Tk()
root.title("Audio Classification")
root.geometry("1340x720+0+0")

# Start and stop control variables
is_running = False
audio_stream = None

# Custom deserialization function for GlorotUniform
def custom_glorot_uniform(shape, dtype=None, seed=None):
    return GlorotUniform(seed=seed)(shape, dtype=dtype)

# Custom deserialization function for Zeros
def custom_zeros(shape, dtype=None):
    return Zeros()(shape, dtype=dtype)

# Load the pre-trained model with custom object scope
custom_objects = {
    'GlorotUniform': custom_glorot_uniform,
    'Zeros': custom_zeros
}
model = load_model(r'C:\Users\Music-Genre-Classification\models\custom_cnn_2d_78.h5', custom_objects=custom_objects)

# Placeholder for predictions
pred = np.zeros(8)

def start():
    global is_running, audio_stream
    if not is_running:
        is_running = True
        audio_stream.start_stream()

def stop():
    global is_running, audio_stream
    if is_running:
        is_running = False
        audio_stream.stop_stream()

# Set up the frames
frame1 = tk.Frame(root, padx=10, pady=10)
frame2 = tk.LabelFrame(root, padx=10, pady=10)

# Set up the first plot
f1 = Figure(figsize=(6, 8), dpi=90)
a = f1.add_subplot()

def animate1(i):
    xValue = pred
    a.clear()
    a.barh([1, 2, 3, 4, 5, 6, 7, 8], xValue, align='center',
           tick_label=['B/C', 'Classical', 'Disco', 'HipHop', 'Jazz', 'Pop', 'Reggae', 'Rock/Metal'])
    for i in range(8):
        a.text(s=str(xValue[i])[:4], x=xValue[i], y=i + 1)
    a.axes.get_xaxis().set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)

canvas1 = FigureCanvasTkAgg(f1, frame1)
canvas1.draw()
canvas1.get_tk_widget().pack()

# Set up the second plot
f2 = Figure(figsize=(9, 8), dpi=90)
ax = f2.add_subplot(2, 1, 1)
ax1 = f2.add_subplot(2, 1, 2)

def animate2(i):
    if not is_running:
        return
    chunk = 33000
    rate = 22050
    data = audio_stream.read(chunk, exception_on_overflow=False)
    data_float = np.frombuffer(data, dtype=np.float32)
    x = librosa.feature.melspectrogram(y=data_float, sr=rate, n_fft=1024, hop_length=256, n_mels=128)
    global pred
    x = x.reshape(1, 128, 129, 1)
    dopreds(x, model)
    ax.clear()
    ax.plot(data_float)
    ax.set_ylim([-1, 1])

def dopreds(x, model):
    global pred
    preds = model.predict(x)
    preds = np.array([preds[0,0] + preds[0,2], preds[0,1], preds[0,3], preds[0,4], preds[0,5], preds[0,7], preds[0,8], preds[0,9] + preds[0,6]])
    pred = (preds + (3 * pred)) / 4

canvas2 = FigureCanvasTkAgg(f2, frame2)
canvas2.draw()
canvas2.get_tk_widget().pack()

# Set up the frames
frame1.grid(row=0, column=0)
frame2.grid(row=0, column=1)

# Set up the audio stream
p = pyaudio.PyAudio()
chosen_device_index = -1
for x in range(p.get_device_count()):
    info = p.get_device_info_by_index(x)
    if info['name'] == 'pulse':
        chosen_device_index = info['index']
        break

audio_stream = p.open(format=pyaudio.paFloat32,
                      channels=1,
                      rate=22050,
                      input=True,
                      output=True,
                      input_device_index=chosen_device_index,
                      frames_per_buffer=33000)

# Create the start and stop buttons
start_button = tk.Button(root, text="Start", command=start)
start_button.grid(row=1, column=0)

stop_button = tk.Button(root, text="Stop", command=stop)
stop_button.grid(row=1, column=1)

# Set up the animations
ani1 = animation.FuncAnimation(f1, animate1, interval=1500)
ani2 = animation.FuncAnimation(f2, animate2)

# Run the main loop
root.mainloop()

# Cleanup
audio_stream.stop_stream()
audio_stream.close()
p.terminate()
