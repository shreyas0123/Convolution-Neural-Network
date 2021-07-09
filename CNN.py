###########################Problem 1######################################
#importing required packages
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout

#defining model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
#Using regularization technique for improved model --dropout technique--
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

#loading dataset
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
#spltiing dataset into train and test
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#compling model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#fitting model to train data
model.fit(train_images, train_labels, epochs=5, batch_size=64)
#calculating testing accuracy and loss
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
test_acc

##########################################Problem 2###########################
