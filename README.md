# MLstripes4JWST

## Description

This repository contains code to train a neural network to remove 1/f striped noise from JWST images. I also provide a set of optimized weights for the model, which can be found under `scripts/weights`. The demo.ipynb file shows how to load the optimized weights and apply the Unet model to predict a clean template of the striped noise, which can then be used to subtract from the original image.


## Getting started

To run demo.ipynb, the following packages along with their dependencies are required:

- numpy: `pip install numpy`.
- matplotlib: `pip install matplotlib`.
- jupyter notebook: `pip install notebook`.
- astropy: `pip install astropy`.
- tensorflow: `pip install tensorflow`.
  - If you encounter issues on macOS with the installation of tensorflow, you can instead try the following steps:

    ```
    conda create --name mlstripes python=3.9
    conda install -c apple tensorflow-deps
    python3 -m pip install tensorflow-macos
    ```
  - You can find more information on installing tensorflow for macOS [here](https://developer.apple.com/metal/tensorflow-plugin/).


If you would like to train your own model using the provided scripts, then you also need to install:

- tensorflow addons: `pip install tensorflow-addons`.
- colorednoise: `pip install colorednoise`.

These additional packages are used to simulate training data and to perform data augmentation.

## Contact

If you have any questions, feel free to send me an email at `ronan.legin@umontreal.ca` .
