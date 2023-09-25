from plotly.tools import mpl_to_plotly
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

loaded_model = tf.keras.models.load_model("./assets/trained_convolutional_80.keras")

_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)


rows = 5
columns = 10
total_images = rows * columns
figure, image_array = plt.subplots(rows, columns)

for row in range(rows):
    for column in range(columns):
        image_array[row, column].imshow(x_test[row * columns + column])

        # remove ticks from primary axes
        image_array[row, column].set_xticks([])
        image_array[row, column].set_yticks([])

        # remove ticks from secondary x axis
        secax = image_array[row, column].secondary_xaxis('top')
        secax.set_xticks([])

        secax.set_xlabel('asd')
        image_array[row, column].set_xlabel("asdasd")

        print(x_test[row * columns + column].shape)

        


plt.show()

prediction = loaded_model.predict(x_test)
print(prediction)


