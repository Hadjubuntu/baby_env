import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import time

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

# Generate fake image
image = np.random.rand(1, 20, 20, 1).astype(np.float32)

# Make Gaussian Kernel with desired specs.
gauss_kernel = gaussian_kernel(size=1, mean=0.0, std=1.0)

# Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

# Convolve.
t = time.time()
x = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

with tf.Session() as sess:
    img_convolved = np.squeeze(x[0, ...].eval())

    dt_tf = time.time() - t
    print(f"tf in {dt_tf} seconds")

    img_src = np.squeeze(image[0, ...])


    imgsrc = Image.fromarray(np.uint8(img_src * 255) , 'L')
    #img.show()
    imgsrc.save('./img_src.bmp')

    img = Image.fromarray(np.uint8(img_convolved * 255) , 'L')
    #img.show()
    img.save('./img_convovled.bmp')


    t = time.time()
    img_scipyfilter = gaussian_filter(img_src, sigma=1.0)
    dt_scipy = time.time() - t
    print(f"scipy blur in {dt_scipy} seconds")


    img = Image.fromarray(np.uint8(img_scipyfilter * 255) , 'L')
    #img.show()
    img.save('./img_scipy.bmp')