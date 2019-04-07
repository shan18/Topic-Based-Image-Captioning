from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, Dropout

from models.embedding_layer import create_embedding_layer


def image_encoder(feature_shape, state_size):
    """ Encode Images """
    feature_input = Input(
        shape=feature_shape, name='feature_input'
    )
    feature_net = Dropout(0.5)(feature_input)
    image_model_output = Dense(state_size, activation='relu', name='image_model_output')(feature_net)
    return feature_input, image_model_output


def caption_encoder(
    topic_input_shape, word_index, glove_file, mark_start, mark_end, state_size, vocab_size, max_tokens
):
    """ Encode Captions """

    # Define layers
    topic_input = Input(
        shape=topic_input_shape, name='topic_input'
    )
    caption_input = Input(shape=(max_tokens,), name='caption_input')
    caption_embedding = create_embedding_layer(word_index, glove_file, mark_start, mark_end, vocab_size)
    caption_lstm = LSTM(state_size, name='caption_lstm')

    # connect layers
    topic_input_reshaped = Reshape(target_shape=(K.int_shape(topic_input)[1:] + (1,)))(topic_input)
    _, initial_state_h0, initial_state_c0 = LSTM(
        state_size, return_state=True, name='topic_lstm'
    )(topic_input_reshaped)
    topic_lstm_states = [initial_state_h0, initial_state_c0]
    net = caption_input  # Start the decoder-network with its input-layer
    net = caption_embedding(net)  # Connect the embedding-layer
    net = Dropout(0.5)(net)
    caption_model_output = caption_lstm(net, initial_state=topic_lstm_states) # Connect the caption LSTM layer

    return topic_input, caption_input, caption_model_output
