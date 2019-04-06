""" This file downloads the VGG19 model pre-trained
    on the imagenet dataset.
    It then removes the last layer from the model
    and returns the modified model.
"""


from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19


def load_vgg19():
    model = VGG19(include_top=True, weights='imagenet')
    conv_layer = model.get_layer('flatten')
    conv_model = Model(inputs=model.input, outputs=conv_layer.output)
    return conv_model
