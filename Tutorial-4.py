# %%
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time
import pickle

NAME = f'Cats-vs-dog-cnn-64x2-NO-DENSE-{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

X = pickle.load(open('models/T2/X.pickle', 'rb'))
Y = pickle.load(open('models/T2/y.pickle', 'rb'))

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X,Y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard])


