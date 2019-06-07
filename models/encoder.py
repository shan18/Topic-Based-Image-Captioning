import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, Dropout

from models.embeddings import create_embedding_layer


def create_image_encoder(feature_model, state_size, dropout):
    """ Encode Images """
    feature_input = Input(
        shape=K.int_shape(feature_model.output)[1:], name='feature_input'
    )
    feature_net = Dropout(dropout)(feature_input)
    image_model_output = Dense(state_size, activation='relu', name='image_model_output')(feature_net)
    return feature_input, image_model_output


def create_caption_encoder(
    topic_model, word_idx, glove_file, mark_start, mark_end, state_size, dropout, vocab_size, max_tokens
):
    """ Encode Captions """

    # Define layers
    topic_input = Input(
        shape=K.int_shape(topic_model.output)[1:], name='topic_input'
    )
    caption_input = Input(shape=(max_tokens,), name='caption_input')
    caption_embedding = create_embedding_layer(word_idx, glove_file, mark_start, mark_end, vocab_size)
    caption_lstm = LSTM(state_size, name='caption_lstm')

    # connect layers
    topic_input_reshaped = Reshape(target_shape=(K.int_shape(topic_input)[1:] + (1,)))(topic_input)
    _, initial_state_h0, initial_state_c0 = LSTM(
        state_size, return_state=True, name='topic_lstm'
    )(topic_input_reshaped)
    topic_lstm_states = [initial_state_h0, initial_state_c0]
    net = caption_input  # Start the decoder-network with its input-layer
    net = caption_embedding(net)  # Connect the embedding-layer
    net = Dropout(dropout)(net)
    caption_model_output = caption_lstm(net, initial_state=topic_lstm_states) # Connect the caption LSTM layer

    return topic_input, caption_input, caption_model_output
