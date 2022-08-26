# MLstripes

## Description

This repository contains code to train a neural network to remove 1/f striped noise from JWST images. I also provide a set of optimized weights for the model, which can be found under `scripts/weights`. The demo.ipynb file shows how to load the optimized weights and apply the Unet model to predict a clean template of the striped noise, which can then be used to subtract from the original image.


## Getting started

To run demo.ipynb, the following dependencies are required:

- numpy: `pip install numpy`.
- tensorflow: `pip install tensorflow`.
- astropy: `pip install astropy`.

If you would like to train your own model using the provided scripts, then you also need to install:

- tensorflow_addons: `pip install tensorflow_addons`.
- colorednoise: `pip install colorednoise`.

These packages are used to simulate training data and to perform data augmentation.

## Contact

If you have any questions, you can send me an email at `ronan.legin@umontreal.ca` .
