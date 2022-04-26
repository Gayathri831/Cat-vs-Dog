#!/usr/bin/env python
# coding: utf-8

# In[96]:


import tensorflow as tf


# In[97]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[125]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'C:/Users/gayat/Downloads/DATA SETS/CNN Cats vs Dogs/training_set/training_set',
        target_size=(150,150),
        batch_size=32,
        class_mode='binary')


# In[126]:


test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        'C:/Users/gayat/Downloads/DATA SETS/CNN Cats vs Dogs/test_set/test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


# In[127]:


#building CNN model
#initializing
cnn = tf.keras.models.Sequential()


# In[128]:


#convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu',input_shape = [150,150,3] ))


# In[129]:


#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# In[130]:


#Adding one more layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# In[131]:


#flatten
cnn.add(tf.keras.layers.Flatten())


# In[132]:


#full connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# In[133]:


#output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[134]:


#compile the model
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])


# In[144]:


#Train the model
history = cnn.fit(x = train_generator, validation_data= validation_generator, epochs=5)


# In[145]:


import numpy as np
from keras_preprocessing import image
test_image = image.load_img('C:/Users/gayat/Downloads/DATA SETS/CNN Cats vs Dogs/sample prediction/1.jpg', target_size=(150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
train_generator.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


# In[146]:


prediction


# In[147]:


import numpy as np
from keras_preprocessing import image
test_image = image.load_img('C:/Users/gayat/Downloads/DATA SETS/CNN Cats vs Dogs/sample prediction/2.jpg', target_size=(150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
train_generator.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


# In[148]:


prediction


# In[149]:


import matplotlib.pyplot as plt
print(history.history.keys())


# In[154]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])


# In[ ]:




