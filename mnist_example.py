import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

# model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# split data betweed train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# scale images to [0,1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# ensure image shape is (28, 28, 1)
print("x_train shape:", x_train.shape)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# build the model
model = keras.Sequential(
    [
        keras.Input(shape = input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation = 'softmax'),
    ]
)

model.summary()

# Train the model
batch_size = 128
epochs = 15
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, \
            batch_size=batch_size, \
            #steps_per_epoch = np.ceil(len(x_train)/batch_size), \
            epochs=epochs, \
            validation_split=0.1)

# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', score[0])
print('test accuracy:', score[1])
