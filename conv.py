import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


rows = 5
columns = 10
total_images = rows * columns
figure, image_array = plt.subplots(rows, columns)

for row in range(rows):
    for column in range(columns):
        image_array[row, column].imshow(x_train[row * columns + column])

plt.show()

# 32 x 32, RGB
input_shape = (32, 32, 3)
classes = 10

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

weight_initializer = tf.keras.initializers.GlorotNormal(seed=781015)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),

        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomContrast(factor=0.25),
        tf.keras.layers.RandomBrightness(factor=0.25),
        tf.keras.layers.RandomRotation(factor=0.1),

        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        # tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(classes, activation="softmax"),
    ]
)

model.summary()

y_train = tf.keras.utils.to_categorical(y_train, classes)
y_test = tf.keras.utils.to_categorical(y_test, classes)

batch_size = 64
epochs = 200

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25, callbacks=[callback])

model.save(filepath="trained_convolutional.keras")

loaded_model = tf.keras.models.load_model("trained_convolutional.keras")

predictions = loaded_model.evaluate(x_test, y_test)
print(predictions[0])
print(predictions[1])
