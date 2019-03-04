from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam


def load_vgg19():
    """ Download VGG19 model and extract the second last fully connected layer
    """
    model = VGG19(include_top=True, weights='imagenet')
    conv_layer = model.get_layer('fc2')
    conv_model = Model(inputs=model.input, outputs=conv_layer.output)
    return conv_model


def create_topic_model(num_classes):
    """ Use pre-trained vgg19 model and add a custom classification layer """

    conv_model = load_vgg19()  # Load VGG19 model
    image_model = Sequential()  # Start a new Keras Sequential model
    image_model.add(conv_model)  # Add VGG19 model
    image_model.add(Dense(num_classes, activation='sigmoid'))  # Add the final classification layer

    # Set the VGG19 layers to be non-trainable
    conv_model.trainable = False
    for layer in conv_model.layers:
        layer.trainable = False

    # Compile the model
    optimizer = Adam(lr=1e-3)
    image_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return image_model


def load_topic_model(num_classes, weights_path):
    """ Load topic model with pre-trained weights """

    model = create_topic_model(num_classes)

    try:
        model.load_weights(weights_path)
        print('Weights loaded.')
    except Exception as e:
        print('Error trying to load weights.')
        
    return model
