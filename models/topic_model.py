from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from models.inception_v3 import load_inception_v3


def create_topic_model(input_shape, output_dim):
    """ Use pre-trained InceptionV3 model and add a custom classification layer """

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
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    return model


def load_topic_model(input_shape, output_dim, weights_path):
    """ Load topic model with pre-trained weights """

    model = create_topic_model(input_shape, output_dim)

    try:
        model.load_weights(weights_path)
        print('Weights loaded.')
    except Exception as e:
        print('Error trying to load weights.')
        print(e)
        
    return model
