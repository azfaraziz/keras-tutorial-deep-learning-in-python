'''
Step 3: Import libraries and modules
'''
# Numpy
import numpy as np
np.random.seed(123)

# Keras model module
from keras.models import Sequential

# Keras core layers
from keras.layers import Dense, Dropout, Activation, Flatten

# Keras CNN layers
from keras.layers import Convolution2D, MaxPooling2D

# Utilities
from keras.utils import np_utils

'''
Step 4: Load image data from MNIST
'''
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print X_train.shape
# (60000, 28, 28)
#60k samples where each image is 28x28

from matplotlib import pyplot as plt
plt.imshow(X_train[0])

