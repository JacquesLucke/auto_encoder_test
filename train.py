import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pprint import pprint
from pathlib import Path
import plotly.express as px
from tensorflow import keras
from tensorflow.keras import layers

real_images_path = Path("./real_images")
images_as_numpy_path = Path("./real_images.npy")

if not images_as_numpy_path.exists():
    image_paths = os.listdir(real_images_path)
    images_data = np.array(
        [np.array(Image.open(real_images_path / path)) for path in image_paths]
    )
    images_data = images_data / 128 - 0.5
    np.save(images_as_numpy_path, images_data)

images_data = np.load(images_as_numpy_path)
image_shape = images_data.shape[1:]
pixel_count = image_shape[0] * image_shape[1]
latent_space_size = pixel_count

encoder = keras.models.Sequential(
    [
        layers.Reshape([*image_shape, 1]),
        layers.Conv2D(filters=3, kernel_size=3),
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(2, activation="tanh"),
        layers.BatchNormalization(),
    ]
)

decoder = keras.models.Sequential(
    [
        layers.Dense(50, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(300, activation="relu"),
        layers.Dense(pixel_count, activation="linear"),
        layers.Reshape(image_shape),
    ]
)

encoder_and_decoder = keras.models.Sequential(
    [
        encoder,
        decoder,
    ]
)


encoder_and_decoder.compile(
    optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError()
)

encoder_and_decoder.fit(x=images_data, y=images_data, epochs=100)

encoder.save("encoder_model")
decoder.save("decoder_model")

for i in range(2):
    image = images_data[i]
    latent = encoder(np.array([image]))[0]
    print(latent)
    result = decoder(np.array([latent]))[0]

    fig = px.imshow(image)
    fig.show()
    fig = px.imshow(result)
    fig.show()
