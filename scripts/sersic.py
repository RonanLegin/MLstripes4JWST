import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# r_ein, elp_xl, elp_yl, xl, yl, gamma_1, gamma_2, r_eff, elp_xs, elp_ys, xs, ys, n 
MU = tf.constant([[0.2, 0.2, 0.0, 2.5]])
HALF_WIDTH = tf.constant([[0.1, 0.2, 0.4, 1.5]])
LOW = MU - HALF_WIDTH
HIGH = MU + HALF_WIDTH

prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=LOW, high=HIGH))



class SersicProfile():
    """Class for generating Sersic sources."""
    def __init__(self,
                img_side = 7.68,
                pix_side = 192
                ):
        
        self.img_side = img_side
        self.pix_side = pix_side


        x = tf.linspace(-1.0, 1.0, self.pix_side) * self.img_side/2
        y = tf.linspace(-1.0, 1.0, self.pix_side) * self.img_side/2
        Xsrc_grid, Ysrc_grid = tf.meshgrid(x, y)

        self.Xsrc_grid = tf.reshape(Xsrc_grid, [-1, self.pix_side, self.pix_side, 1])
        self.Ysrc_grid = tf.reshape(Ysrc_grid, [-1, self.pix_side, self.pix_side, 1])


    def generate_source(self, params, peak_intensity = 1.0):

        params = tf.reshape(params, [-1, params.shape[-1], 1, 1])
        r_eff, elp_x, elp_y, n = tf.split(params, num_or_size_splits=4, axis=1)

        elp = tf.sqrt(elp_x**2 + elp_y**2)
        elp_angle = tf.atan2(elp_y, elp_x) * 180./np.pi

        x = tf.cos(elp_angle * np.pi/180.)*self.Xsrc_grid + tf.sin(elp_angle * np.pi/180.)*self.Ysrc_grid
        y = -tf.sin(elp_angle * np.pi/180.)*self.Xsrc_grid + tf.cos(elp_angle * np.pi/180.)*self.Ysrc_grid

        R = tf.sqrt(x**2 + (1 - elp)*y**2)

        bn = 2*n -1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)
        
        source = peak_intensity * tf.exp(-bn * ((R/r_eff)**(1/n) - 1))
        
        return source
