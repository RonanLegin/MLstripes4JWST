import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import colorednoise as cn

def sample_stripes(npix, nsamples):

    # Sample horizontal pink noise stripes with variable amplitude
    h_amp = np.random.normal(loc=0.04, scale=0.01, size=(nsamples, 1))
    h_pink1d = h_amp * cn.powerlaw_psd_gaussian(1, [nsamples, npix])
    h_pink2d = np.tile(h_pink1d[:, :, np.newaxis, np.newaxis], [1, 1, npix, 1])

    # Sample vertical pink noise stripes with variable amplitude
    v_amp = np.random.normal(loc=0.005, scale=0.005)
    v_pink1d = v_amp * cn.powerlaw_psd_gaussian(1, [nsamples, 1, npix])
    v_pink2d = np.tile(v_pink1d[:, :, :, np.newaxis], [1, npix, 1, 1])

    # Add random bias
    bias = np.random.normal(loc=0.20, scale=0.02, size=(nsamples, 1, 1, 1))

    hv_pink2d = h_pink2d + v_pink2d + bias

    return hv_pink2d


# Generate noise with horizontal and vertical stripes
def sample_noise(npix, nsamples):

    # Generate 4 quadrants with different striped noise
    hv_pink2d = np.zeros((nsamples, npix, npix, 1), dtype=np.float32)
    hv_pink2d[:, :npix//2, :npix//2] = sample_stripes(npix//2, nsamples)
    hv_pink2d[:, :npix//2, npix//2:] = sample_stripes(npix//2, nsamples)
    hv_pink2d[:, npix//2:, :npix//2] = sample_stripes(npix//2, nsamples)
    hv_pink2d[:, npix//2:, npix//2:] = sample_stripes(npix//2, nsamples)

    # Add bad columns
    num_columns = np.random.randint(0, 5)
    columns = np.random.randint(0, hv_pink2d.shape[2], size=num_columns)
    bad_shift = np.random.uniform(-0.8, 0.8, size=num_columns)
    for c, shift in zip(columns, bad_shift):
        column_width = np.random.choice([1,3,5])
        hv_pink2d[:,:,c - column_width: c + column_width] += shift

    # Sample white noise with variable amplitude
    white_amp = np.random.normal(loc=0.1, scale=0.01, size=(nsamples, 1, 1, 1))
    white2d = white_amp * np.random.normal(size=(nsamples, npix, npix, 1))

    noise = hv_pink2d + white2d

    return noise, hv_pink2d


# Generate a mask for a simulated dead pixel
def sample_dead_pixel_mask(npix, nsamples, ndead=100):

    prob_dead = np.random.randint(0, npix**2, size=(nsamples, npix, npix, 1))
    mask = tf.cast(prob_dead < npix**2 - ndead, dtype=tf.float32)
    amp = tf.random.uniform([nsamples, 1, 1, 1], -1.0, 0.)
    
    return mask, amp

# Generate a mask for a simulated circle region of dead pixels
def sample_circle_mask(npix, nsamples):

    Y, X = np.ogrid[:npix, :npix]
    dist_from_center = np.sqrt((X - npix//2)**2 + (Y-npix//2)**2)

    radius = np.random.randint(5, 10)
    mask = tf.cast(dist_from_center > radius, dtype=tf.float32)

    mask = mask[tf.newaxis,:,:,tf.newaxis]
    mask = tf.tile(mask, [nsamples, 1, 1, 1])

    shift = tf.random.uniform([nsamples,2], minval=-npix//2, maxval=npix//2, dtype=tf.float32)
    mask = tfa.image.translate(mask, shift, fill_mode='wrap') # Randomly shift images
    amp = tf.random.uniform([nsamples, 1, 1, 1], -1.0, 0.)
    
    return mask, amp


# Generate a mask for a simulated cosmic ray
def sample_cosmic_ray_mask(npix, nsamples):

    cr_length = np.random.randint(5, 50)
    cr_width = np.random.randint(3,6)

    mask = np.ones((nsamples, npix, npix, 1), dtype=np.float32)
    mask[:,npix//2 - cr_width:npix//2 + cr_width, npix//2 - cr_length:npix//2 + cr_length] = 0.
    mask = tf.cast(mask, dtype=tf.float32)

    angle = tf.random.uniform([nsamples], minval=0., maxval=2*np.pi, dtype=tf.float32)
    mask = tfa.image.rotate(mask, angle, fill_mode='constant', fill_value=1.0)

    shift = tf.random.uniform([nsamples,2], minval=-npix//2, maxval=npix//2, dtype=tf.float32)
    mask = tfa.image.translate(mask, shift, fill_mode='wrap') # Randomly shift images
    amp = tf.random.uniform([nsamples, 1, 1, 1], -1.0, 0.)

    return mask, amp


# Apply mask to images and replace by constant amplitude
def apply_mask_amp(img, mask, amp):
    # Apply mask to set masked regions to 0
    img *= mask
    # Flip the mask regions from 0 to 1, multiply by amplitude, and add to the masked image
    img += -1. * amp * (mask - 1.)
    return img


# Apply mask to images
def apply_mask(img, mask):
    img *= mask
    return img


