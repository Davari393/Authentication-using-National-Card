from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


np.random.seed(75)

# Reshape to original image shape (n x 784)  ==> (n x 28 x 28 x 1)
x_train = r'dataset\train'
x_test = r'dataset\test'
x_validation = r'dataset\validation'



#define model architecture
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(25, 25, 3)))
model.add(Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))


model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        #train_dir
         x_train,
        # All images will be resized to 150x150
        target_size=(25, 25),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


validation_generator = test_datagen.flow_from_directory(
        #validation_dir
        x_validation,
        target_size=(25, 25),
        batch_size=10,
        class_mode='categorical')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=60,
      epochs=60,
      validation_data=validation_generator,
      validation_steps=20)

test_generator = test_datagen.flow_from_directory(
        # This is the target directory
        x_test,
        # All images will be resized to 150x150
        target_size=(25, 25),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
eval=model.evaluate_generator(test_generator, 200)
print(eval)
model.save('National_Card.h5')