import numpy as np 


# Load COSMOS galaxy images for training, validation and testing

num_per_file = 1000

def load_training_data(path):

    imgs = np.zeros((45*num_per_file, 192, 192), dtype=np.float32)
    for i in np.arange(0, 45):
        imgs[num_per_file*i:num_per_file*i+num_per_file] = np.load(path + 'src_{}.npy'.format((i+1)*num_per_file))
    imgs = imgs[...,np.newaxis]

    return imgs


def load_validation_data(path):

    imgs = np.zeros((1*num_per_file, 192, 192), dtype=np.float32)

    for i in np.arange(0, 1):
        imgs[num_per_file*i:(num_per_file*i+num_per_file)] = np.load(path + 'src_{}.npy'.format((i+1)*num_per_file + 45000))
    imgs = imgs[...,np.newaxis]

    return imgs


def load_testing_data(path):

    imgs = np.zeros((1*num_per_file, 192, 192), dtype=np.float32)
    for i in np.arange(0, 1):
        imgs[num_per_file*i:(num_per_file*i+num_per_file)] = np.load(path + 'src_{}.npy'.format((i+1)*num_per_file  + 46000))
    imgs = imgs[...,np.newaxis]

    return imgs
