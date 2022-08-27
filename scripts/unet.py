"""Define u-net."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose


tf.keras.backend.set_floatx('float32')


class Unet(tf.keras.Model):

    def __init__(self, stacks, stackwidth, filters_base, kernel_size, output_act='tanh', strides=2, filters_mult=2, save_path = 'unet/', **kw):
        """
        Parameters
        ----------
        stacks : int
            Number of unet stacks.
        stackwidth : int
            Number of convolutions per stack.
        filters_base : int
            Base number of intermediate convolution filters.
        output_channels : int
            Number of output channels
        kernel_size : int or tuple of ints
            Kernel size for each dimension, default: (3,3)
        kernel_center : int or tuple of ints
            Kernel center for each dimension, default: kernel_size//2
        strides : tuple of ints
            Strides for each dimension, default: (2,2)
        filters_mult: int
            Multiplicative increase to filter size per stack depth

        Other keyword arguments are passed to the convolution layers.

        """
        super().__init__()

        # Save attributes
        self.stacks = stacks
        self.stackwidth = stackwidth
        self.filters_base = filters_base
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters_mult = filters_mult
        self.conv_kw = kw
        self.save_path = save_path
        self.output_act = output_act

        # Downward stacks, downsampling at the beginning of each except the first
        self.down_stacks = []
        for i in range(stacks):
            stack = []
            filters = filters_base * filters_mult**i
            # Downsampling
            if i != 0:
                stack.append(Conv2D(filters, kernel_size, strides=strides, **kw))
            # Convolutions
            for j in range(stackwidth):
                stack.append(Conv2D(filters, kernel_size, **kw))
            self.down_stacks.append(stack)

        # Upward stacks, upsampling at the end of each except the last
        self.up_stacks = []
        for i in reversed(range(stacks)):
            stack = []
            filters = filters_base * filters_mult**i
            # Convolutions
            for j in range(stackwidth):
                stack.append(Conv2D(filters, kernel_size, **kw))
            # Upsampling
            if i != 0:
                stack.append(Conv2DTranspose(filters//filters_mult, kernel_size, strides=strides, **kw))
            self.up_stacks.append(stack)

        # Output layer
        self.outlayer = Conv2D(1, (1, 1), padding='same', activation=output_act)

    def call(self, x):
        """Call unet on multiple inputs, combining at bottom."""

        def eval_down(x):
            """Evaluate down unet stacks, saving partials."""
            partials = []
            for stack in self.down_stacks:
                # Apply stack
                for layer in stack:
                    x = layer(x)
                    
                # Save partials
                partials.append(x)
            return partials

        def eval_up(partials):
            """Evaluate up unet stacks, concatenating partials."""
            x = None
            for stack in self.up_stacks:
                if x is None:
                    # Start with last partial
                    x = partials.pop()
                else:
                    # Concatenate partial
                    x = tf.concat([x, partials.pop()], axis=3)
                # Apply stack
                for layer in stack:
                    x = layer(x)
            return x
        
        partials = eval_down(x)
        # Evaluate up
        x = eval_up(partials)
        # Apply output layer
        return self.outlayer(x)

