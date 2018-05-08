
# coding: utf-8

# ### Step 3: Import libraries and modules

# In[ ]:


# Numpy
import numpy as np
np.random.seed(123)


# In[ ]:


# Keras model module
from keras.models import Sequential


# In[ ]:


# Keras core layers
from keras.layers import Dense, Dropout, Activation, Flatten


# In[ ]:


# Keras CNN layers
from keras.layers import Conv2D, MaxPooling2D


# In[ ]:


# Utilities
from keras.utils import np_utils


# ### Step 4: Load image data from MNIST

# In[ ]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


print X_train.shape
# (60000, 28, 28)
#60k samples where each image is 28x28


# In[ ]:


from matplotlib import pyplot as plt
plt.imshow(X_train[0])


# ### Step 5 Preprocess input data for Keras

# In[ ]:


from keras import backend as K
print K.image_data_format()


# In[ ]:


# Since the image_data_format() is "channels_last", reshape with channel as the last index
# Convert from shape (n, width, height) to (n, depth, width, height)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print X_train.shape


# In[ ]:


# Convert datatype to float32 and normalize the data to range [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# ### Step 6: Preprocess class labels for Keras

# In[ ]:


# (60000,)
print y_train.shape


# In[ ]:


# Convert single list to 2D (example, class label)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# In[ ]:


print Y_train.shape


# In[ ]:


# Declare Sequential model
model = Sequential()


# In[ ]:


# CNN input layer
'''
input shape should be the shape of 1 sample
The first 3 numbers refer to 
1) number of convolution filters to use, 
2) the number of rows in each convolution kernel, 
3) the number of columns in each convolution kernel
'''
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))


# In[ ]:


# (None, 32, 26, 26)
print model.output_shape


# In[ ]:


# Layering the model
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# In[ ]:


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# ### Step 8: Compile model

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Step 9: Fit model on training data

# In[ ]:


model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)


# In[ ]:


score = model.evaluate(X_test, Y_test, verbose=1)


# In[ ]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])

