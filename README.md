# MLstripes4JWST

## Description

This repository contains code to train a neural network to remove 1/f striped noise from JWST images. The demo.ipynb file shows how to load the optimized weights and apply the Unet model to predict a clean template of the striped noise, which can then be used to subtract from the original image.


## Getting started

To run demo.ipynb, the following packages along with their dependencies are required:

- numpy: `pip install numpy`.
- matplotlib: `pip install matplotlib`.
- jupyter notebook: `pip install notebook`.
- astropy: `pip install astropy`.
- tensorflow: `pip install tensorflow`.
  - If you encounter issues on macOS with the installation of tensorflow, you can instead follow the instructions [here](https://developer.apple.com/metal/tensorflow-plugin/). Specifically, if you have a Mac with the M1 chip, I would try creating a conda environment and installing all the packages in the following order:

    ```
    conda create --name mlstripes python=3.9
    conda activate mlstripes
    pip install numpy
    pip install matplotlib
    pip install jupyter notebook
    pip install astropy
    conda install -c apple tensorflow-deps
    python3 -m pip install tensorflow-macos
    ```

If you would like to train your own model using the provided scripts, then you also need to install:

- tensorflow addons: `pip install tensorflow-addons`.
- colorednoise: `pip install colorednoise`.

These additional packages are used to simulate training data and to perform data augmentation.

## Weights

Since the network's weights exceeded the file size limit on Github, I uploaded the weights [here](https://drive.google.com/drive/folders/1m9SHNbJDbocI4unNYFzq2h1m1uFst1JV?usp=sharing). To run demo.ipynb with these weights, simply drag and drop the weights folder in 'scripts'.

## Contact

If you have any questions, feel free to send me an email at `ronan.legin@umontreal.ca` .
