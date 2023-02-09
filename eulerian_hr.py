
"""eulerian_hr.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from google.colab import drive
drive.mount('/content/drive')

import os
path = 'drive/My Drive'

from google.colab.patches import cv2_imshow

import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Read in and simultaneously preprocess video
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []
    face_rects = ()
    i=-1
    while cap.isOpened():
        ret, img = cap.read()
        i+=1
        if i%4!=0:
          continue
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roi_frame = img

        # Detect face
        if len(video_frames) == 0:
            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Select ROI
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                roi_frame = img[y:y + h, x:x + w]
            if roi_frame.size != img.size:
                roi_frame = cv2.resize(roi_frame, (500, 500))
                frame = np.ndarray(shape=roi_frame.shape, dtype="float")
                frame[:] = roi_frame * (1. / 255)
                video_frames.append(frame)

    #video_frames = video_frames[::5]
    frame_ct = len(video_frames)
    cap.release()

    return video_frames, frame_ct, fps

import cv2
import numpy as np


# Build Gaussian image pyramid
def build_gaussian_pyramid(img, levels):
    float_img = np.ndarray(shape=img.shape, dtype="float")
    float_img[:] = img
    pyramid = [float_img]

    for i in range(levels-1):
        float_img = cv2.pyrDown(float_img)
        pyramid.append(float_img)

    return pyramid


# Build Laplacian image pyramid from Gaussian pyramid
def build_laplacian_pyramid(img, levels):
    gaussian_pyramid = build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []

    for i in range(levels-1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1])
        (height, width, depth) = upsampled.shape
        gaussian_pyramid[i] = cv2.resize(gaussian_pyramid[i], (height, width))
        diff = cv2.subtract(gaussian_pyramid[i],upsampled)
        laplacian_pyramid.append(diff)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


# Build video pyramid by building Laplacian pyramid for each frame
def build_video_pyramid(frames):
    lap_video = []

    for i, frame in enumerate(frames):
        pyramid = build_laplacian_pyramid(frame, 3)
        for j in range(3):
            if i == 0:
                lap_video.append(np.zeros((len(frames), pyramid[j].shape[0], pyramid[j].shape[1], 3)))
            lap_video[j][i] = pyramid[j]

    return lap_video


# Collapse video pyramid by collapsing each frame's Laplacian pyramid
def collapse_laplacian_video_pyramid(video, frame_ct):
    collapsed_video = []

    for i in range(frame_ct):
        prev_frame = video[-1][i]

        for level in range(len(video) - 1, 0, -1):
            pyr_up_frame = cv2.pyrUp(prev_frame)
            (height, width, depth) = pyr_up_frame.shape
            prev_level_frame = video[level - 1][i]
            prev_level_frame = cv2.resize(prev_level_frame, (height, width))
            prev_frame = pyr_up_frame + prev_level_frame

        # Normalize pixel values
        min_val = min(0.0, prev_frame.min())
        prev_frame = prev_frame + min_val
        max_val = max(1.0, prev_frame.max())
        prev_frame = prev_frame / max_val
        prev_frame = prev_frame * 255

        prev_frame = cv2.convertScaleAbs(prev_frame)
        collapsed_video.append(prev_frame)

    return collapsed_video

from scipy import signal


# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []
    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60

import numpy as np
import scipy.fftpack as fftpack


# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(video, freq_min, freq_max, fps):
    fft = fftpack.fft(video, axis=0)
    frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)
    result *= 100  # Amplification factor

    return result, fft, frequencies

def calculate(name):
  # Frequency range for Fast-Fourier Transform
  freq_min = 0.8
  freq_max = 2.0

  # Preprocessing phase
  print("Reading + preprocessing video...")
  video_frames, frame_ct, fps = read_video(f'drive/My Drive/{name}/vid.avi')
  print("frames", len(video_frames))
  # Build Laplacian video pyramid
  lap_video = build_video_pyramid(video_frames)

  amplified_video_pyramid = []
  for i, video in enumerate(lap_video):
      if i == 0 or i == len(lap_video)-1:
          continue

      # Eulerian magnification with temporal FFT filtering
      result, fft, frequencies = fft_filter(video, freq_min, freq_max, fps)
      lap_video[i] += result

      # Calculate heart rate
      heart_rate = find_heart_rate(fft, frequencies, freq_min, freq_max)
      print("Heart rate: ", heart_rate, "bpm")

  # # Collapse laplacian pyramid to generate final video
  # print("Rebuilding final video...")
  # amplified_frames = collapse_laplacian_video_pyramid(lap_video, frame_ct)
  return heart_rate


  # for frame in amplified_frames:
  #     cv2_imshow(frame)
  #     cv2.waitKey(20)

# Test the function
dirs = os.listdir('drive/My Drive')
predicted = [94.4186046511628, 108.02660753880265, 66.14035087719299, 80.72164948453607, 96.28458498023716, 93.21428571428571, 99.72332015810277, 51.377952755905504, 90.84337349397592, 58.57425742574259, 73.22645290581163, 103.36633663366338, 50.87719298245614, 98.5546875, 99.13555992141454, 55.238095238095234, 85.62992125984252, 65.14285714285714, 110.6958250497018]
for sub in range(23):
  sub+=25
  video = f'subject{sub}'
  if(dirs.count(video)==0):
    continue
  print(video)
  hr = calculate(video)
  predicted.append(hr)

gt = os.path.join('drive/My Drive/subject5/ground_truth.txt')


gtfilename = gt

gtdata = np.loadtxt(gtfilename)
gtTrace = gtdata[0,:].T
gtTime = gtdata[2,:].T
gtHR = gtdata[1,:].T
gtHR = gtHR[::10]
#print(gtHR)
print(sum(gtHR)/len(gtHR))
# the variable `img` now contains the image data as a 2D or 3D numpy array

dirs = os.listdir('drive/My Drive')
actual = []
for sub in range(48):
  sub+=1
  video = f'subject{sub}'
  if(dirs.count(video)==0):
    continue
  gt = os.path.join('drive/My Drive',video,'ground_truth.txt')


  gtfilename = gt

  gtdata = np.loadtxt(gtfilename)
  gtTrace = gtdata[0,:].T
  gtTime = gtdata[2,:].T
  gtHR = gtdata[1,:].T
  gtHR = gtHR[::12]
  #print(gtHR)
  av = sum(gtHR)/len(gtHR)
  actual.append(av)
mae =0;
mae2 = 0
print(predicted)
c=0
for i in range(len(predicted)):
  print(f'{predicted[i]}  {actual[i]}')
  if actual[i]>=111 or predicted[i]<62:
    mae2 = mae2+abs(predicted[i]-actual[i])
    continue
  c+=1
  mae = mae+abs(predicted[i]-actual[i])

mae/=c
mae2/=(len(predicted)-c)
print(mae)
print(mae2)

print(predicted)
print(c)
print(len(predicted))
