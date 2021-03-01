import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from os.path import isfile
from scipy.special import binom

# Grayscale representation
GRAYSCALE = 1
# RGB representation
RGB = 2
# Number of grayscales
BINS = 256
# Minimum Dimention
MIN_DIM = 16


# Reads filename to an image (numpy array) and return the image according to the given representation.
# filename - the filename of an image on disk (could be grayscale or RGB).
# representation - representation code, either 1 or 2 defining whether the output should be a grayscale
# image (1) or an RGB image (2). If the input image is grayscale.
def read_image(filename, representation):
    if not isfile(filename):
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


# im – a grayscale image with double values in [0, 1]
# (e.g. the output of ex1’s read_image with the representation set to 1).
# max_levels – the maximal number of levels1 in the resulting pyramid.
# filter_size – the size of the Gaussian filter (an odd scalar that represents a squared filter)
# to be used in constructing the pyramid filter
# (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
# return - A tuple (pyr, filter_vec).
# pyr - the resulting pyramid as a standard python array (NOT numpy)
# filter_vec - a normalized row vector of shape (1, filter_size) used for the pyramid construction
def build_gaussian_pyramid(im, max_levels, filter_size):
    # Binomial row vec for the filter
    filter_vec = np.array([binom(filter_size - 1, i) for i in range(filter_size)]).reshape((1, filter_size))
    filter_vec *= 1 / np.power((filter_size - 1), 2)  # Normalization
    im_i = im
    pyr = [im_i]
    for i in range(1, max_levels):
        min_dim = min(im_i.shape)
        if min_dim <= MIN_DIM:
            break
        im_i = downsample_image(im_i, filter_vec)
        pyr.append(im_i)
    return pyr, filter_vec


# Down sample an image.
# Taking even indices every even row.
# image - one channel image to down sample
# filter_vec - filter vector of shape (1, num)
# return - downsamples image
def downsample_image(image, filter_vec):
    # Blur
    blurred_image = blur_image(image, filter_vec)
    # Sub-sample: Taking pixel of even row at even indices
    ds_image = blurred_image[::2, ::2]
    return ds_image


# Expand an image.
# expanded image new shape is (2n, 2m), except odd indices at odd rows.
# image - one channel image to down sample
# filter_vec - filter vector of shape (1, num)
# return - expanded image
def expand_image(image, filter_vec):
    n, m = image.shape
    # Expanded zeros array of shape (2n, 2m)
    expanded_image = np.zeros((2 * n, 2 * m), dtype=image.dtype)
    # Pad zeros. Taking pixel of odd row at odd indices
    expanded_image[1::2, 1::2] = image
    # Blur
    return blur_image(expanded_image, filter_vec)


# Blur a given image with filter_vec using 2D-convolution.
# image - one channel image to down sample
# filter_vec - filter vector of shape (1, num)
# return - blurred image
def blur_image(image, filter_vec):
    blurred_im = convolve2d(image, filter_vec, mode='same')  # Conv with row vec
    blurred_im = convolve2d(blurred_im, filter_vec.reshape((filter_vec.size, 1)), mode='same')  # Conv with col vec
    return blurred_im


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


# Check if image has rgb channels
# image - image as np array.
def rgb_check(image):
    return len(image.shape) == 3 and image.shape[2] == 3


# Check if image has grayscale channel
# image - image as np array.
def grayscale_check(image):
    return len(image.shape) == 2
