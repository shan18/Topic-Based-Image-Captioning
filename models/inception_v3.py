""" This file downloads the InceptionV3 model pre-trained
    on the imagenet dataset.
    It then removes the last layer from the model
    and returns the modified model.
"""


from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3


def load_inception_v3():
    model = InceptionV3(include_top=True, weights='imagenet')
    conv_model = Model(model.input, model.layers[-2].output)
    return conv_model
