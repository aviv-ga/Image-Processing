import numpy as np
import matplotlib.pyplot as plt
import os.path
from imageio import imread
from skimage.color import rgb2gray

# number of bins
BINS = 255
# RGB to YIQ matrix transformation
RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]], dtype=np.float64)
# YIQ to RGB matrix transformation
YIQ2RGB = np.linalg.inv(RGB2YIQ)
# RGB representation
RGB = 2
# Grayscale representation
GRAYSCALE = 1


# Reads filename to an image (numpy array) and return the image according to the given representation.
# filename - the filename of an image on disk (could be grayscale or RGB).
# representation - representation code, either 1 or 2 defining whether the output should be a grayscale
# image (1) or an RGB image (2). If the input image is grayscale.
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


# Displays an image.
# filename - the filename of an image on disk (could be grayscale or RGB).
# representation - representation code, either 1 or 2 defining whether the output should be a grayscale
# image (1) or an RGB image (2). If the input image is grayscale.
def imdisplay(filename, representation):
    im = read_image(filename, representation)
    if im is None:
        return None
    if representation == GRAYSCALE:
        plt.imshow(im, cmap="gray")
    elif representation == RGB:
        plt.imshow(im)
    plt.show()


# transform an RGB image into the YIQ color space.
# imRGB - image in rgb color space with shape of (height, width, channels)
# return - image in YIQ color space with same dimensions as the input.
def rgb2yiq(imRGB):
    return imRGB @ np.transpose(RGB2YIQ)


# transform an YIQ image into the RGB color space.
# imYIQ - image in YIQ color space with shape of (height, width, channels)
# return - image in RGB color space with same dimensions as the input.
def yiq2rgb(imYIQ):
    rgb = imYIQ @ np.transpose(YIQ2RGB)
    return rgb.clip(0, 1)


# Check if image has rgb channels
# image - image as np array.
def rgb_check(image):
    return len(image.shape) == 3 and image.shape[2] == 3


# Check if image has grayscale channel
# image - image as np array.
def grayscale_check(image):
    return len(image.shape) == 2


# Histogram equalization:
# im_orig - the input grayscale or RGB float64 image with values in [0, 1].
# The function returns a list [im_eq, hist_orig, hist_eq] where
# im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
# hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
# hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
def histogram_equalize(im_orig):
    if rgb_check(im_orig):
        yiq = rgb2yiq(im_orig)
        im_eq, hist_orig, hist_eq = histogram_helper(yiq[:, :, 0])  # only Y channel
        yiq[:, :, 0] = im_eq  # set Y channel to the new equalized image
        im_eq = np.clip(yiq2rgb(yiq), 0, 1)  # go back to rgb and clip
        return [im_eq, hist_orig, hist_eq]
    elif grayscale_check(im_orig):
        return histogram_helper(im_orig)
    else:
        return None


# Helper function of histogram_equalize.
# image - An image with only 1 channel with shape of (x,y)
# Return - A list: [im_eq, hist_orig, hist_eq] where:
# im_eq is equalized image, hist_orig is the histogram of the original image,
# and hist_eq is the histogram of the equalized image.
def histogram_helper(image):
    height, width = image.shape
    hist_orig = np.histogram(image, bins=256, range=[0, 1])[0]
    cum_hist = np.cumsum(hist_orig)
    norm_cum_hist = cum_hist.astype(np.float64) / (height * width)
    norm_cum_hist *= BINS
    stretched_cum_hist = norm_cum_hist - norm_cum_hist.min()  # shift left
    stretched_cum_hist *= (BINS / stretched_cum_hist.max())  # stretch
    stretched_cum_hist = stretched_cum_hist.astype(np.uint8)  # round
    im_eq = stretched_cum_hist[(image * BINS).astype(np.uint8)].astype(np.float64)
    im_eq /= BINS
    hist_eq = np.histogram(im_eq, bins=256)[0]
    return [im_eq, hist_orig, hist_eq]


# quantization procedure
# im_orig - is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
# n_quant - is the number of intensities your output im_quant image should have.
# n_iter - is the maximum number of iterations of the optimization procedure (may converge earlier.)
# Return - a list [im_quant, error] where:
# im_quant - is the quantized output image.
# error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
def quantize(im_orig, n_quant, n_iter):
    if grayscale_check(im_orig):
        return quantize_helper(im_orig, n_quant, n_iter)
    if rgb_check(im_orig):
        yiq = rgb2yiq(im_orig)
        out = quantize_helper(yiq[:, :, 0], n_quant, n_iter)
        yiq[:, :, 0] = out[0]
        return [yiq2rgb(yiq), out[1]]


def quantize_helper(image, n_quant, n_iter):
    image *= (BINS / image.max())
    hist, bins = np.histogram(image, bins=BINS + 1, range=[0, 255])
    z_0 = initiate_z(n_quant, hist, image.size)  # initial division
    z, q, error = quantize_main(hist, bins, z_0, n_quant, n_iter)  # the heart of quantization
    mapped_image = map_q(image, q, z, n_quant)  # map q to segments
    return mapped_image, error


# Computing initial z - the borders which divide the histograms into segments. z is an array with shape
# (n_quant+1,). The first and last elements are 0 and 255 respectively
# n_quant - number of segments.
# hist - the histogram to quantize
# return - z, an np array with shape (n_quant+1,) containing segments borders
def initiate_z(n_quant, hist, size):
    cum_hist = np.cumsum(hist)
    z = np.empty(n_quant + 1, dtype=np.uint8)
    z[0] = 0  # leftmost border
    z[n_quant] = BINS  # rightmost border
    seg_average = size / n_quant  # number of pixels for each segment
    for i in range(1, n_quant):
        z[i] = np.argmin(np.abs(cum_hist - (i * seg_average)))
    return z


# Main calculation for quantization process.
# hist - Histogram of current image to be processed
# bins - bins of the histogram
# z - Initial borders which divide the histogram into segments. (np array of shape (n_quant+1,))
# n_quant - The number of intensities your output im_quant image should have.
# n_iter - is the maximum number of iterations of the optimization procedure.
def quantize_main(hist, bins, z, n_quant, n_iter):
    error = []
    total_error = 0
    q = np.empty(n_quant, dtype=np.float64)
    for i in range(n_iter):
        pixels_error = np.empty(n_quant)
        for j in range(n_quant):
            seg_j = bins[z[j]:z[j + 1] + 1]
            p_z = hist[z[j]:z[j + 1] + 1]
            p_z = p_z / np.sum(p_z)
            q[j] = np.dot(p_z, seg_j)
            q_j = np.full(seg_j.size, q[j])  # q_j is an array of the calculated q[j]
            pixels_error[j] = np.dot(np.power(q_j - seg_j, 2), p_z)
        if total_error == np.sum(pixels_error):  # in case of early convergence
            return z, q, error
        total_error = np.sum(pixels_error)
        error.append(total_error)
        for j in range(1, n_quant):
            z[j] = ((q[j - 1] + q[j]) / 2).round()
    return z, q, error


# map segments to calculated q values.
# image - the preprocessed image.
# q - the intensities for each segment.
# z - the borders which divide the histogram into segments. (np array of shape (n_quant+1,))
# n_quant - The number of intensities the output im_quant image should have.
# return - mapped image
def map_q(image, q, z, n_quant):
    mapped_image = np.empty(BINS + 1, dtype=np.float64)
    for i in range(n_quant):
        mapped_image[z[i]:z[i + 1]] = q[i]  # z[i]:z[i+1] = segment_i
    return mapped_image[image.astype(np.uint8)] / BINS
