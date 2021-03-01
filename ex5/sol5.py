import sol5_utils
import numpy as np
import random
from os.path import isfile
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from keras.layers import Input, Activation, Conv2D, Add
from keras import Model
from keras.optimizers import Adam


# Grayscale representation
GRAYSCALE = 1
# RGB representation
RGB = 2
# Number of grayscales
BINS = 256
# Used for normal distribution for gaussian noise.
MIN_SIGMA = 0
MAX_SIGMA = 0.2
# Patch sizes for denoising and deblurring model
DENOISING_PATCH = 24
DEBLURRING_PATCH = 16
# Number of output channels for denosing and deblurring
DENOSING_OUTPUT_CHANNEL = 48
DEBLURRING_OUTPUT_CHANNEL = 32
# Kernel for 2D Convolution
KERNEL = (3, 3)


# filenames – A list of filenames of clean images.
# batch_size – The size of the batch of images for each iteration of Stochastic Gradient Descent.
# corruption_func – A function receiving a numpy’s array representation of an image as a single argument,
# and returns a randomly corrupted version of the input image.
# crop_size – A tuple (height, width) specifying the crop size of the patches to extract
def load_dataset(filenames, batch_size, corruption_func, crop_size):
    crop_height, crop_width = crop_size
    cache = {}

    while True:
        source_batch = []
        target_batch = []
        for i in range(batch_size):
            name = random.choice(filenames)
            if name in cache.keys():
                im = cache[name]
            else:
                im = read_image(name, GRAYSCALE)
                cache[name] = im

            # Generate large patch and corrupt
            large_patch = crop_image(im, (3 * crop_height, 3 * crop_width))[0]
            corrupted_large_patch = corruption_func(large_patch)
            # Generate patch and corrupted patch
            patch, dx, dy = crop_image(large_patch, crop_size)
            corrupted_patch = crop_image(corrupted_large_patch, crop_size, dx, dy)[0]
            # Range is now [-0.5, 0.5]
            patch -= 0.5
            corrupted_patch -= 0.5
            # add generated data to source and target batches
            target_batch.append(patch)
            source_batch.append(corrupted_patch)

        # make it Numpy array
        out_shape = (batch_size, crop_height, crop_width, 1)
        source_batch = np.asarray(source_batch).reshape(out_shape)
        target_batch = np.asarray(target_batch).reshape(out_shape)
        yield source_batch, target_batch


# Crop image.
# im - image to crop.
# crop size - tuple of (height, width)
# dx, dy - if mentioned crop accordingly to their values, otherwise
# generate random values for them.
def crop_image(im, crop_size, dx=-1, dy=-1):
    if dx == -1:
        dx = random.randint(0, im.shape[0] - crop_size[0])
        dy = random.randint(0, im.shape[1] - crop_size[1])
    return im[dx:dx + crop_size[0], dy: dy + crop_size[1]], dx, dy


# Define a block for our Neural Network:
# Block scheme: Conv2D -> ReLuU -> Conv2D -> Addition(with skip connection) -> ReLuU
# Conv2D arguments: output channels, (window’s height, window’s width)
# Activation argument: specifying activation type
# input_tensor - nd numpy's array
# num_channels - number of channels to be returned by Conv2D
def resblock(input_tensor, num_channels):
    x = Conv2D(num_channels, KERNEL, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(num_channels, KERNEL, padding='same')(x)
    x = Add()([input_tensor, x])
    x = Activation('relu')(x)
    return x


# Build ResNet
# height - input height
# width - input width
# num_channels - number of output channel of conv2D operation.
# num_res_blocks - number of resBlocks in NN.
def build_nn_model(height, width, num_channels, num_res_blocks):
    inp = Input(shape=(height, width, 1))
    x = Conv2D(num_channels, KERNEL, padding='same')(inp)
    x = Activation('relu')(x)
    # Connect ResBlocks
    for i in range(num_res_blocks):
        x = resblock(x, num_channels)
    x = Conv2D(1, KERNEL, padding='same')(x)
    out = Add()([inp, x])
    return Model(inputs=inp, outputs=out)


# model – ResNet, a neural network model for image restoration.
# images – a list of file paths pointing to image files. You should assume these paths are complete, and
# should append anything to them.
# corruption_func – A function receiving a numpy’s array representation of an image as a single argument,
# # and returns a randomly corrupted version of the input image.
# batch_size – the size of the batch of examples for each iteration of SGD.
# steps_per_epoch – The number of update steps in each epoch.
# num_epochs – The number of epochs for which the optimization will run.
# num_valid_samples – The number of samples in the validation set to test on after every epoch.
def train_model(model, images, corruption_func, batch_size,
                steps_per_epoch, num_epochs, num_valid_samples):
    random.shuffle(images)
    train_size = np.int(len(images) * 0.8)
    train_set = images[:train_size]
    validation_set = images[train_size:]
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))

    crop_size = model.input_shape[1:3]
    train_generator = load_dataset(train_set, batch_size, corruption_func, crop_size)
    valid_generator = load_dataset(validation_set, batch_size, corruption_func, crop_size)
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=valid_generator, validation_steps=num_valid_samples)


# corrupted_image – a grayscale image of shape (height, width) and with values in the [0, 1] range of
# type float64
# base_model – a neural network trained to restore small patches.
# The input and output of the network are images with values in the [−0.5, 0.5] range.
def restore_image(corrupted_image, base_model):
    h, w = corrupted_image.shape
    a = Input(shape=(h, w, 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    # Reshaped to: 1 image of shape=(h, w), with 1 channel
    reshaped_corrupted_im = corrupted_image.reshape(1, h, w, 1) - 0.5
    restored_image = new_model.predict(reshaped_corrupted_im).reshape(h, w)
    return np.clip(restored_image + 0.5, 0, 1)


# Add Gaussian noise to an image.
# sigma is uniformly distributed between min_sigma and max_sigma.
# followed by adding to every pixel of the input image a zero-mean gaussian random variable
# with standard deviation equal to sigma
# image - a grayscale image with values in the [0, 1] range of type float64.
# Min_sigma - a non-negative scalar value representing the minimal variance of the gaussian distribution.
# max_sigma – a non-negative scalar value larger than or equal to min_sigma, representing the maximal
# variance of the gaussian distribution.
def add_gaussian_noise(image, min_sigma, max_sigma):
    # min_sigma < rand_sigma < max_sigma
    rand_sigma = np.random.uniform(min_sigma, max_sigma)
    # Gaussian noise of shape im.shape
    noise = np.random.normal(loc=0, scale=rand_sigma, size=image.shape)
    noised_image = image + noise
    noised_image = np.round(noised_image * BINS) / BINS
    return np.clip(noised_image, 0, 1)


# Learn denoising model
# Num_res_blocks - number of resBlocks in NN.
# quick_mode - if true: batch size = 10
#                       steps_per_epoch = 3
#                       epochs = 2
#                       num_valid_samples = 30
# quick_mode - if false: batch size = 100
#                       steps_per_epoch = 100
#                       epochs = 5
#                       num_valid_samples = 1000
# return - learned model
def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    return learn_model("denoising", num_res_blocks, quick_mode)


# image – a grayscale image with values in the [0, 1] range of type float64.
# kernel_size – an odd integer specifying the size of the kernel (even integers are ill-defined).
# angle – an angle in radians in the range [0, π).
def add_motion_blur(image, kernel_size, angle):
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


# image – a grayscale image with values in the [0, 1] range of type float64.
# list_of_kernel_sizes – a list of odd integers.
def random_motion_blur(image, list_of_kernel_sizes):
    angle = random.uniform(0, np.pi)
    kernel_size = random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)


# Learn a deblurring model.
# Num_res_blocks - number of resBlocks in NN.
# quick_mode - if true: batch size = 10
#                       steps_per_epoch = 3
#                       epochs = 2
#                       num_valid_samples = 30
# quick_mode - if false: batch size = 100
#                       steps_per_epoch = 100
#                       epochs = 5
#                       num_valid_samples = 1000
# return - learned model.
def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    return learn_model("deblurring", num_res_blocks, quick_mode)


# Learn a model according to paramater learning.
# learning - "denoising" or "deblurring"
# Num_res_blocks - number of resBlocks in NN.
# quick_mode - if true: batch size = 10
#                       steps_per_epoch = 3
#                       epochs = 2
#                       num_valid_samples = 30
# quick_mode - if false: batch size = 100
#                       steps_per_epoch = 100
#                       epochs = 5
#                       num_valid_samples = 1000
# return a learned model
def learn_model(learning, num_res_blocks=5, quick_mode=False):
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        epochs = 2
        num_valid_samples = 30
    else:
        batch_size = 100
        steps_per_epoch = 100
        epochs = 5
        num_valid_samples = 1000

    if learning == "denoising":
        model = build_nn_model(DENOISING_PATCH, DENOISING_PATCH, DENOSING_OUTPUT_CHANNEL, num_res_blocks)
        images = sol5_utils.images_for_denoising()
        corruption_func = lambda image: add_gaussian_noise(image, MIN_SIGMA, MAX_SIGMA)
    elif learning == "deblurring":
        model = build_nn_model(DEBLURRING_PATCH, DEBLURRING_PATCH, DEBLURRING_OUTPUT_CHANNEL, num_res_blocks)
        images = sol5_utils.images_for_deblurring()
        corruption_func = lambda image: random_motion_blur(image, [7])
    else:
        return

    train_model(model, images, corruption_func, batch_size, steps_per_epoch, epochs, num_valid_samples)
    # The model is now trained
    return model


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


# Check if image has rgb channels
# image - image as np array.
def rgb_check(image):
    return len(image.shape) == 3 and image.shape[2] == 3


# Check if image has grayscale channel
# image - image as np array.
def grayscale_check(image):
    return len(image.shape) == 2
