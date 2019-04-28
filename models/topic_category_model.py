from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from models.vgg19 import load_vgg19


# def create_category_model(num_classes):
#     """ Use pre-trained vgg19 model and add a custom classification layer """

#     conv_model = load_vgg19()  # Load VGG19 model
#     image_model = Sequential()  # Start a new Keras Sequential model
#     image_model.add(conv_model)  # Add VGG19 model
#     image_model.add(Dense(num_classes, activation='sigmoid'))  # Add the final classification layer

#     # Set the VGG19 layers to be non-trainable
#     conv_model.trainable = False
#     for layer in conv_model.layers:
#         layer.trainable = False

#     # Compile the model
#     optimizer = Adam(lr=1e-3)
#     image_model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=['binary_accuracy']
#     )

#     return image_model


# def load_category_model(num_classes, weights_path):
#     """ Load topic model with pre-trained weights """

#     model = create_category_model(num_classes)

#     try:
#         model.load_weights(weights_path)
#         print('Weights loaded.')
#     except:
#         print('Error trying to load weights.')
        
#     return model


def create_category_model(input_shape, output_dim):
    """ Use pre-trained vgg19 model and add a custom classification layer """

    feature_input = Input(
        shape=input_shape, name='feature_input'
    )
    topic_output = Dense(output_dim, activation='sigmoid')(feature_input)  # Add the final classification layer

    # Define model
    model = Model(
        inputs=feature_input,
        outputs=topic_output
    )

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def load_category_model(input_shape, output_dim, weights_path):
    """ Load topic model with pre-trained weights """

    model = create_category_model(input_shape, output_dim)

    try:
        model.load_weights(weights_path)
        print('Weights loaded.')
    except:
        print('Error trying to load weights.')
        
    return model
