#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


# In[2]:


def get_data(filename):
  # You will need to write code that will read the file passed
  # into this function. The first line contains the column headers
  # so you should ignore it
  # Each successive line contians 785 comma separated values between 0 and 255
  # The first value is the label
  # The rest are the pixel values for that picture
  # The function will return 2 np.array types. One with all the labels
  # One with all the images
  #
  # Tips: 
  # If you read a full line (as 'row') then row[0] has the label
  # and row[1:785] has the 784 pixel values
  # Take a look at np.array_split to turn the 784 pixels into 28x28
  # You are reading in strings, but need the values to be floats
  # Check out np.array().astype for a conversion
    images, labels = list(), list()
    with open(filename) as training_file:
      # Your code starts here
        reader = csv.reader(training_file, delimiter=',')
        for row in reader:
            if row[0] == 'label': continue
            lab = np.asarray(row[0], np.int16)
            img = np.asarray(row[1:], dtype=np.int16)
            img = np.stack(np.array_split(img, 28))
            images.append(img)
            labels.append(lab)
    images = np.asarray(images)
    labels = np.asarray(labels)
      # Your code ends here
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)


# In[3]:


# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims


def _to_categorical(values):
    n = np.max(values) + 1
    return np.eye(n)[values]

training_images = np.expand_dims(training_images, -1)
testing_images  = np.expand_dims(testing_images, -1)

y_train = _to_categorical(training_labels)
y_test  = _to_categorical(testing_labels)  


# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow(training_images, y_train)
validation_generator = validation_datagen.flow(testing_images, y_test)
    
# Keep These
print(training_images.shape)
print(testing_images.shape)
    
# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)


# In[4]:


# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D

l = tf.keras.layers

model = tf.keras.models.Sequential([
    l.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    l.MaxPool2D(),
    l.Conv2D(64, 3, activation='relu'),
#     l.MaxPool2D(),
    l.Flatten(),
    l.Dense(256, activation='relu'),
    l.Dense(64, activation='relu'),
    l.Dense(25, activation='softmax')]
)

# Compile Model. 
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

# Train the Model
history = model.fit_generator(train_generator, epochs=15, validation_data = validation_generator, verbose = 1)

model.evaluate(testing_images, y_test, verbose=0)


# In[5]:


# Plot the chart for accuracy and loss on both training and validation
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

print(set(history.history.keys()))

h = history.history

acc = h['accuracy']
val_acc = h['val_accuracy']
loss = h['loss']
val_loss = h['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # Submission Instructions

# In[ ]:


# Now click the 'Submit Assignment' button above.


# # When you're done or would like to take a break, please run the two cells below to save your work and close the Notebook. This will free up resources for your fellow learners. 

# In[ ]:


get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.session.delete();\nwindow.onbeforeunload = null\nsetTimeout(function() { window.close(); }, 1000);')

