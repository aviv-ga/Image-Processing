import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
from scipy.special import binom
from imageio import imread
from skimage.color import rgb2gray


# Grayscale representation
GRAYSCALE = 1
# RGB representation
RGB = 2
# Number of grayscales
BINS = 256
# Minimum Dimention
MIN_DIM = 16


def check_par(im, max_levels, filter_size):
    # check we get a grayscale image
    if rgb_check(im):
        return False
    # max_levels is at least 1 or max log2 im size
    if max_levels < 1:
        return False
    min_dim = min(im.shape)
    i = 0
    new_max_levels = max_levels
    while min_dim / np.power(2, max_levels - i) < MIN_DIM:
        new_max_levels -= 1
    # odd and positive value for filter_size
    if not filter_size % 2:
        return False
    return new_max_levels


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
    new_max_levels = check_par(im, max_levels, filter_size)
    # True if new_max_lvl is zero
    if not new_max_levels:
        return None  # invalid input

    # Binomial row vec for the filter
    filter_vec = np.array([binom(filter_size - 1, i) for i in range(filter_size)]).reshape((1, filter_size))
    filter_vec *= 1 / np.power((filter_size - 1), 2)  # Normalization
    im_i = im
    pyr = [im_i]
    for i in range(1, new_max_levels):
        im_i = downsample_image(im_i, filter_vec)
        pyr.append(im_i)
    return pyr, filter_vec


# im – a grayscale image with double values in [0, 1]
# (e.g. the output of ex1’s read_image with the representation set to 1).
# max_levels – the maximal number of levels1 in the resulting pyramid.
# filter_size – the size of the Gaussian filter (an odd scalar that represents a squared filter)
# to be used in constructing the pyramid filter
# (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
# return - A tuple (pyr, filter_vec).
# pyr - the resulting pyramid as a standard python array (NOT numpy)
# filter_vec - a normalized row vector of shape (1, filter_size) used for the pyramid construction
def build_laplacian_pyramid(im, max_levels, filter_size):
    new_max_levels = check_par(im, max_levels, filter_size)
    # True if new_max_lvl is zero
    if not new_max_levels:
        return None  # invalid input

    g_pyr, filter_vec = build_gaussian_pyramid(im, new_max_levels, filter_size)
    filter_vec *= 2  # maintain constant brightness
    expanded_g = []
    for i in range(1, new_max_levels):
        expanded_g.append(expand_image(g_pyr[i], filter_vec))
    lap_pyr = [g_pyr[i] - expanded_g[i] for i in range(new_max_levels - 1)]
    lap_pyr.append(g_pyr[-1])
    return lap_pyr, filter_vec


# Reconstructing an image from its Laplacian Pyramid
# lpyr - Laplacian pyramid.
# filter_vec - filter vector of shape (1, n)
# coeff - python list. The list length is the same as the number of levels in the pyramid lpyr.
# return - the reconstructed image.
def laplacian_to_image(lpyr, filter_vec, coeff):
    # Expand each level of the pyramid to its
    img = lpyr[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        img = expand_image(img * coeff[i], filter_vec) + lpyr[i - 1]
    return img


# pyr - either a Gaussian or Laplacian pyramid
# levels - the number of levels to present in the result ≤ max_levels.
# return - a single black image in which the pyramid levels of the given pyramid pyr
# are stacked horizontally.
def render_pyramid(pyr, levels):
    # Stretch first level image
    res = (pyr[0] - (pyr[0]).min()) / (pyr[0]).max()
    for im in pyr[1:levels]:
        # Stretch
        res_i = (im - np.min(im)) / np.max(im)
        # Create black background
        black_background = np.zeros((pyr[0].shape[0] - im.shape[0], im.shape[1]))
        # Stack black background to smaller image
        res_i = np.vstack((res_i, black_background))
        # stack res_i to the first level image
        res = np.hstack((res, res_i))
    return res


# Display a pyramid
# pyr - either a Gaussian or Laplacian pyramid
# levels - the number of levels to present in the result ≤ max_levels.
def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()


# Blend 2 images
# im1, im2 – are two input grayscale images to be blended
# mask - a boolean mask containing True and False representing which parts
# of im1 and im2 should appear in the resulting image.
# max_levels - max levels of the Gaussian and Laplacian pyramids.
# filter_size_im – the size of the Gaussian filter (an odd scalar that represents a squared filter) which
# defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
# filter_size_mask – the size of the Gaussian filter (an odd scalar that represents a squared filter) which
# defining the filter used in the construction of the Gaussian pyramid of mask.
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    lpyr_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lpyr_2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    l_out = [(g_mask[i] * lpyr_1[i]) + (1 - g_mask[i]) * lpyr_2[i] for i in range(max_levels)]
    im_blend = laplacian_to_image(l_out, filter_vec, list(np.ones((max_levels,))))
    return im_blend.clip(0, 1)


# Blending example - 1
def blending_example1():
    im1 = read_image(relpath('externals/basketball.jpg'), 2)
    im2 = read_image(relpath('externals/football.jpg'), 2)
    mask = (np.arange(1024)[:, None] + np.arange(1024)) % 2 == 0
    blended_im = blend_rgb(im1, im2, mask, 6, 5, 3)
    example_display(im1, im2, mask, blended_im, 1)
    return im1, im2, mask, blended_im


# Blending example - 2
def blending_example2():
    im1 = read_image(relpath('externals/chinesewall.jpg'), 2)
    im2 = read_image(relpath('externals/lionking.jpg'), 2)
    mask = read_image(relpath('externals/mask.jpg'), 1)
    mask = mask.astype(np.bool)
    blended_im = blend_rgb(im1, im2, mask, 4, 5, 3)
    example_display(im1, im2, mask, blended_im, 2)
    return im1, im2, mask, blended_im


# Blend 2 RGB images using pyramid_blending on each channel.
def blend_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    res = np.zeros(im1.shape)
    for i in range(im1.shape[2]):
        res[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size_im, filter_size_mask)
    return res


# Display 4 images in a single figure.
def example_display(im1, im2, mask, blended_im, num):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle('Example ' + str(num), fontsize=20)
    axs[0, 0].imshow(im1)
    axs[0, 1].imshow(im2)
    axs[1, 0].imshow(blended_im)
    axs[1, 1].imshow(mask, cmap='gray')
    plt.show()


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


# filename - image relative path
# return - absolute path to an image.
def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


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


# Check if image has rgb channels
# image - image as np array.
def rgb_check(image):
    return len(image.shape) == 3 and image.shape[2] == 3


# Check if image has grayscale channel
# image - image as np array.
def grayscale_check(image):
    return len(image.shape) == 2

