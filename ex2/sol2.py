import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

# number of bins
BINS = 255
# RGB representation
RGB = 2
# Grayscale representation
GRAYSCALE = 1
# Change rate file name
CHANGE_RATE = "change_rate.wav"
# Change sample file name
CHANGE_SAMPLES = "change_samples.wav"
# Derivative vector for convolution
DERIVATIVE_VEC = np.array([[0.5], [0], [-0.5]])
# 2*pi*i
FT_EXP = 2 * np.pi * complex(0, 1)


def read_image(filename, representation):
    if not os.path.isfile(filename):
        return None
    im = imread(filename)
    im_float = im.astype(np.float64)
    im_float /= BINS
    if representation == GRAYSCALE:
        if rgb_check(im):
            return rgb2gray(im_float)
        else:
            return im_float
    elif representation == RGB:
        if grayscale_check(im_float):
            return None  # should not reach that point
        else:
            return im_float
    else:  # should not reach that point
        return None


# Check if image has rgb channels
# image - image as np array.
def rgb_check(image):
    return len(image.shape) == 3 and image.shape[2] == 3


# Check if image has grayscale channel
# image - image as np array.
def grayscale_check(image):
    return len(image.shape) == 2


# Discrete Fourier Transform
# signal - an array of dtype float64 with shape (N,1).
# return: Fourier signal - an array of dtype complex128 with the same shape (N,1)
def DFT(signal):
    return DFT_basis(signal.shape[0], False) @ signal


# Inverse Discrete Fourier Transform
# Fourier signal - an array of dtype complex128 with shape (N,1).
# return:  signal - an array of dtype float64 with the same shape (N,1)
def IDFT(fourier_signal):
    return DFT_basis(fourier_signal.shape[0], True) @ fourier_signal


# Helper function for DFT and IDFT.
# This function creates the fourier basis (matrix nXn) of the transform.
# n - the size of the matrix.
# inverse - true if inverse basis is required, false otherwise.
# return - nXn matrix of the fourier basis.
def DFT_basis(n, inverse):
    fac = -1
    complex_power = FT_EXP / n
    if inverse:
        fac = 1
    complex_power *= fac
    # col vec @ row vec = matrix
    basis = np.exp(complex_power * (np.arange(n).reshape(n, 1) @ np.arange(n).reshape(1, n)))
    if inverse:
        basis /= n
    return basis


# 2D discrete fourier transform
# convert a 2D discrete signal to its Fourier representation.
# image - a grayscale image of dtype float64
# return - fourier_image, a 2D array of dtype complex128
def DFT2(image):
    height, width = image.shape
    first_trans = DFT_basis(height, False) @ image
    second_trans = DFT_basis(width, False) @ first_trans.T
    return second_trans.T


# 2D discrete inverse fourier transform
# convert a 2D discrete signal in Fourier representation to its signal in the time representation.
# fourier_image - a 2D array of dtype complex128
# return - image, a grayscale image of dtype float64
def IDFT2(fourier_image):
    height, width = fourier_image.shape
    first_inv = DFT_basis(height, True) @ fourier_image
    second_inv = DFT_basis(width, True) @ first_inv.T
    return second_inv.T


# Change the given WAV file frequency according to the given ratio.
# Result will be saved in a file called rate.wav.
# filename - a string representing the path to a WAV file.
# ratio - a positive float64, 0.25 < ratio < 4, representing the duration change.
def change_rate(filename, ratio):
    if ratio < 0.25 or ratio > 4:
        print("Ratio should be 0.25 <= ratio <= 4")
        return None
    sr, data = wavfile.read(filename)
    new_sr = sr * ratio
    wavfile.write(CHANGE_RATE, new_sr, data)


# fast forward function that changes the duration of an audio file by reducing the number of
# samples using Fourier. Result will be saved in a file called change_samples.wav.
# filename - a string representing the path to a WAV file.
# ratio - a positive float64 (0.25 < ratio < 4)representing the duration change.
# return - 1D ndarray of dtype float64 representing the new sample points
def change_samples(filename, ratio):
    if ratio < 0.25 or ratio > 4:
        print("Ratio should be 0.25 <= ratio <= 4")
        return None
    sr, data = wavfile.read(filename)
    new_data = resize(data, ratio)
    wavfile.write(CHANGE_SAMPLES, sr, new_data)


# speeds up a WAV file, without changing the pitch, using spectrogram scaling
# data - a 1D ndarray of dtype float64 representing the original sample points.
# ratio - a positive float64 (0.25 < ratio < 4) representing the duration change.
# return - the new resized data.
def resize_spectrogram(data, ratio):
    spect = stft(data)
    # resize the spectrogram by changing rows's length according to ratio
    resized_spect = np.array([resize(spect[i, :], ratio) for i in range(spect.shape[0])])
    return istft(resized_spect)


# speeds up a WAV file, without changing the pitch, using spectrogram scaling, but includes the
# correction of the phases of each frequency according to the shift of each window.
# data - a 1D ndarray of dtype float64 representing the original sample points.
# ratio - a positive float64 (0.25 < ratio < 4) representing the duration change.
# return - the new resized data.
def resize_vocoder(data, ratio):
    return istft(phase_vocoder(stft(data), ratio))


# Resize a np array of shape (n,) or (n,1) according to the given ratio.
# DFT and IDFT are applied if dtype is not complex128
# data - a 1D ndarray of dtype float64 or complex128 representing the original sample points.
# ratio - a positive float64 (0.25 < ratio < 4)representing the duration change.
# return - a 1D ndarray of the dtype of data representing the new sample points
def resize(data, ratio):
    if ratio < 0.25 or ratio > 4:
        print("Ratio should be 0.25 <= ratio <= 4")
        return None
    if ratio == 1:
        return data
    new_data_size = int(np.floor(data.size / ratio))
    edge = int(np.abs(np.floor((data.size - new_data_size) / 2)))
    # check if DFT is required
    if data.dtype == np.complex128:
        fourier_data = data
    else:
        fourier_data = DFT(data)
    # shift the zero-frequency component to the center of the spectrum
    shifted_fourier_data = np.fft.fftshift(fourier_data)
    if ratio > 1:
        # chop high frequencies both sides
        if np.abs(data.size - new_data_size) % 2:  # odd size of data
            resized_fourier_data = shifted_fourier_data[edge:(data.size - edge - 1)]
        else:  # even size of data
            resized_fourier_data = shifted_fourier_data[edge:(data.size - edge)]

    elif ratio < 1:
        # pad zeros both sides
        if len(data.shape) == 1:
            concat_zeros = np.zeros(edge).reshape((edge,))
        else:
            concat_zeros = np.zeros(edge).reshape((edge, data.shape[1]))
        resized_fourier_data = np.concatenate((concat_zeros, shifted_fourier_data, concat_zeros))

    # shift back
    shifted_back_fourier_data = np.fft.ifftshift(resized_fourier_data)
    # check if IDFT is required
    if data.dtype == np.complex128:
        new_data = shifted_back_fourier_data
    else:
        new_data = IDFT(shifted_back_fourier_data)
    # remove tiny imaginary parts before return
    return new_data


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    # time_steps = np.arange(spec.shape[1]) * ratio
    # time_steps = time_steps[time_steps < spec.shape[1]]
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


# computes the magnitude of image derivatives using convolution.
# im - grayscale image of type float64 with shape (n, m).
# return - magnitude of the derivative
def conv_der(im):
    dx = signal.convolve2d(im, DERIVATIVE_VEC, mode='same')
    dy = signal.convolve2d(im, DERIVATIVE_VEC.T, mode='same')
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


# Computes the magnitude of image derivatives using Fourier.
# im - grayscale image of type float64 with shape (n, m).
# return - magnitude of the derivative
def fourier_der(im):
    height, width = im.shape
    fourier_im = DFT2(im)
    shifted_fourier_im = np.fft.fftshift(fourier_im)
    # Multiply by [-n/2 ... 0 ... n/2]
    dx_fac = np.arange(np.ceil(-width / 2), np.ceil(width / 2), dtype=np.float64).reshape((1, width))
    dy_fac = np.arange(np.ceil(height/2), np.ceil(-height/2), -1, dtype=np.float64).reshape((height, 1))
    # Shift back then apply IDFT2 then multiply by the derivative factor
    dx = (FT_EXP / height) * IDFT2(np.fft.ifftshift(shifted_fourier_im * dx_fac))
    dy = (FT_EXP / width) * IDFT2(np.fft.ifftshift(shifted_fourier_im * dy_fac))
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
