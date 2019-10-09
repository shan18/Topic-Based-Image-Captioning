from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


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
