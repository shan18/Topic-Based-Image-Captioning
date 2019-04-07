from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization


def create_category_model(input_shape, output_dim):
    """ Use pre-trained vgg19 model and add a custom classification layer """

    feature_input = Input(
        shape=input_shape, name='feature_input'
    )
    feature_net = Dense(4096, activation='relu')(feature_input)
    feature_net = Dropout(0.5)(feature_net)
    feature_net = BatchNormalization()(feature_net)
    feature_net = Dense(1000, activation='relu')(feature_input)
    feature_net = Dropout(0.5)(feature_net)
    feature_net = BatchNormalization()(feature_net)
    topic_output = Dense(output_dim, activation='sigmoid')(feature_net)  # Add the final classification layer

    # Define model
    model = Model(
        inputs=feature_input,
        outputs=topic_output
    )

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def load_category_model(input_shape, output_shape, weights_path):
    """ Load topic model with pre-trained weights """

    model = create_category_model(input_shape, output_shape)

    try:
        model.load_weights(weights_path)
        print('Weights loaded.')
    except:
        print('Error trying to load weights.')
        
    return model

