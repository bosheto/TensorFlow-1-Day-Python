
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time
import pickle


dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers = [1,2,3]

X = pickle.load(open('models/T2/X.pickle', 'rb'))
Y = pickle.load(open('models/T2/y.pickle', 'rb'))

X = X/255.0

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}'
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(X,Y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard])



