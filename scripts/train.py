import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from unet import Unet

from parameters import stacks, stackwidth, filters_base, kernel_size, kw, learning_rate, batch_size, num_steps, cosmos_npix
from noise import sample_noise, sample_dead_pixel_mask, sample_circle_mask, sample_cosmic_ray_mask, apply_mask_amp, apply_mask
from load_cosmos import load_training_data, load_validation_data

import sys

# Directory where data folder is contained.
tmp_path = sys.argv[1]


# Define the loss function. Here, we use the mean squared error between the true clean image and the one predicted by the network.
def loss_function(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred)**2)


@tf.function
def train_step(x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model.call(x)  # Call model on the input data
        loss = loss_function(y_true, y_pred)    # Compute loss
    grads = tape.gradient(loss, model.trainable_weights)    # Compute gradient of network weights wrt the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Apply gradients to optimize weights

    return loss


# Same as the train_step function, but here we do not optimize weights
@tf.function
def valid_step(x, y_true):
    y_pred = model.call(x)
    loss = loss_function(y_true, y_pred)
    return loss


# Perform random transformations to the input and target data to create more diversity in the training data.
@tf.function
def augment_data(img):
    shift = tf.random.uniform([batch_size,2], minval=-cosmos_npix, maxval=cosmos_npix, dtype=tf.float32)
    img = tfa.image.translate(img, shift, fill_mode='wrap') # Randomly shift images
    scale = tf.random.uniform([], 1, 3, dtype=tf.int32) # Randomly resize images
    img = tf.image.resize(img, tf.cast([img.shape[1]*scale, img.shape[2]*scale], tf.int32))
    return img


# Multiply image by the amplitude of COSMOS dataset background source. 
# This simulates the fact that after applying a Tanh scaling to real data,
# the noise and stripes at bright sources gets squashed down.
def apply_src_scaling(img, src):
    # Flip src img by switching 0s to 1s and scale img.
    img = - (src - 1.0)*img
    return img


# Applies masks and scales noise before feeding to the neural network
def preprocess_batch(src_batch, noise, pink_template):

    # Sample masked regions for dead pixels and cosmic rays
    mask_dp, amp_dp = sample_dead_pixel_mask(src_batch.shape[1], batch_size)
    mask_cr, amp_cr = sample_cosmic_ray_mask(src_batch.shape[1], batch_size)
    mask_cl, amp_cl = sample_circle_mask(src_batch.shape[1], batch_size)

    # Add together clean image and noise, simulate noise scaling due to tanh preprocessing
    x_batch = src_batch + apply_src_scaling(noise, src_batch)
    y_batch = apply_src_scaling(pink_template, src_batch)

    # Mask images and replace masked regions with constant amplitude value
    x_batch = apply_mask_amp(x_batch, mask_dp, amp_dp)
    x_batch = apply_mask_amp(x_batch, mask_cr, amp_cr)
    x_batch = apply_mask_amp(x_batch, mask_cl, amp_cl)

    # Mask regions in the true templates
    y_batch = apply_mask(y_batch, mask_dp)
    y_batch = apply_mask(y_batch, mask_cr)
    y_batch = apply_mask(y_batch, mask_cl)

    return x_batch, y_batch





# Initialize Unet model 
model = Unet(stacks, stackwidth, filters_base, kernel_size, save_path='weights', **kw)    

# Initialize the type of optimizer and learning rate to perform gradient descent. 
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


src_train = load_training_data(tmp_path + 'COSMOS_23_2/')
src_valid = load_validation_data(tmp_path + 'COSMOS_23_2/')

# Only necessary for my own image set COSMOS for training:
# Crop images to remove empty surrounding space
src_train = src_train[:, 48:-48, 48:-48]
src_valid = src_valid[:, 48:-48, 48:-48]


# Begin training loop

train_loss = []

for i in range(0, num_steps):
  
    # Randomly select a batch of images
    batch_index = np.random.randint(0, src_train.shape[0], batch_size)   
    src_batch = src_train[batch_index]
    
    # Resize COSMOS image data back to original size
    src_batch = tf.image.resize(src_batch, tf.cast([cosmos_npix, cosmos_npix],tf.int32))

    # Randomly modify images to add more variety to the training data
    src_batch = augment_data(src_batch)

    # Sample noise containing horizontal and vertical stripes
    noise, pink_template = sample_noise(src_batch.shape[1], batch_size)

    # Convert Numpy arrays into Tensor arrays
    pink_template = tf.convert_to_tensor(pink_template, dtype=tf.float32)
    src_batch = tf.convert_to_tensor(src_batch, dtype=tf.float32)
    noise = tf.convert_to_tensor(noise, dtype=tf.float32)

    x_batch, y_batch = preprocess_batch(src_batch, noise, pink_template)

    # Perform a training step, which updates the network's weights.
    loss = train_step(x_batch, y_batch)
    train_loss.append(loss)

    
    # After 1000 steps, apply the model on the validation set and record its average loss
    if i % 1000 == 0:
        valid_loss = []

        # Here we give the network one batch of images at a time.
        # We generally try to make the batches as large as possible given memory restrictions.
        for j in range(src_valid.shape[0]//batch_size - 1):

                src_batch = src_valid[j*batch_size:(j+1)*batch_size]

                # Resize COSMOS image data back to original size
                src_batch = tf.image.resize(src_batch, tf.cast([cosmos_npix, cosmos_npix],tf.int32))

                # Randomly modify images to add more variety to the training data
                src_batch = augment_data(src_batch)

                # Sample noise containing horizontal and vertical stripes
                noise, pink_template = sample_noise(src_batch.shape[1], batch_size)

                # Convert Numpy arrays into Tensor arrays
                pink_template = tf.convert_to_tensor(pink_template, dtype=tf.float32)
                src_batch = tf.convert_to_tensor(src_batch, dtype=tf.float32)
                noise = tf.convert_to_tensor(noise, dtype=tf.float32)

                x_batch, y_batch = preprocess_batch(src_batch, noise, pink_template)

                # Perform a training step, which updates the network's weights.
                loss = valid_step(x_batch, y_batch)
                valid_loss.append(loss)


        print('Training step: {}, Training loss: {:.6f}, Validation loss: {:.6f}'.format(i, np.mean(np.array(train_loss)), np.mean(np.array(valid_loss))))

        train_loss = []
        # Save the newly optimized weights of the neural network.
        model.save_weights(model.save_path + '/weights')
        print("Weights saved.")


        




