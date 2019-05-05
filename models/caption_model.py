import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add, Reshape, Dropout

from models.vgg19 import load_vgg19
from models.topic_model import load_topic_model


def load_pre_trained_image_model(weights_path, input_shape, num_classes):
    topic_model = load_topic_model(input_shape, num_classes, weights_path)
    feature_model = load_vgg19()
    print('Done.\n')
    return topic_model, feature_model


def create_image_encoder(feature_model, state_size, dropout):
    """ Encode Images """
    feature_input = Input(
        shape=K.int_shape(feature_model.output)[1:], name='feature_input'
    )
    feature_net = Dropout(dropout)(feature_input)
    image_model_output = Dense(state_size, activation='relu', name='image_model_output')(feature_net)
    return feature_input, image_model_output


def create_word_vec_map(glove_file, mark_start, mark_end):
    print('Creating word to vec map...')
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)

    size = word_to_vec_map['unk'].shape
    word_to_vec_map[mark_start.strip()] = np.random.uniform(low=-1.0, high=1.0, size=size)
    word_to_vec_map[mark_end.strip()] = np.random.uniform(low=-1.0, high=1.0, size=size)
    print('Done!')

    return word_to_vec_map


def create_embedding_layer(word_to_index, glove_file, mark_start, mark_end, num_words):
    """ Create a Keras Embedding() layer and load in pre-trained GloVe 100-dimensional vectors
        @params:
        :word_to_index -- dictionary containing the each word mapped to its index
        :word_to_vec_map -- dictionary mapping words to their GloVe vector representation
        :num_words -- number of words in the vocabulary
        
        @return:
        :decoder_embedding -- pretrained layer Keras instance
    """

    # Create word_vec map
    word_to_vec_map = create_word_vec_map(glove_file, mark_start, mark_end)
    
    vocabulary_length = num_words + 1  # adding 1 to fit Keras embedding (requirement)
    embedding_dimensions = word_to_vec_map['unk'].shape[0]  # define dimensionality of GloVe word vectors (= 300)
    
    embedding_matrix = np.zeros((vocabulary_length, embedding_dimensions))  # initialize with zeros
    for word, index in word_to_index.items():
        try:
            embedding_matrix[index, :] = word_to_vec_map[word]
        except KeyError:
            embedding_matrix[index, :] = word_to_vec_map['unk']
    
    # we don't want the embeddings to be updated, thus trainable parameter is set to False
    decoder_embedding = Embedding(
        input_dim=vocabulary_length,
        output_dim=embedding_dimensions,
        trainable=False,
        name='decoder_embedding'
    )
    decoder_embedding.build((None,))
    decoder_embedding.set_weights([embedding_matrix])  # with this the layer is now pretrained
    
    return decoder_embedding


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


def create_model(
    image_model_weights, feature_input_shape, num_topics, state_size, dropout,
    word_idx, glove_file, mark_start, mark_end, vocab_size, max_tokens=16
):
    # Load pre-trained image model
    topic_model, feature_model = load_pre_trained_image_model(image_model_weights, feature_input_shape, num_topics)

    # Encode Images
    feature_input, image_model_output = create_image_encoder(feature_model, state_size, dropout)

    # Encode Captions
    topic_input, caption_input, caption_model_output = create_caption_encoder(
        topic_model, word_idx, glove_file, mark_start, mark_end, state_size, dropout, vocab_size, max_tokens
    )
    
    # merge encoders and create the decoder
    merge_net = Add()([image_model_output, caption_model_output])
    merge_net = Dense(state_size, activation='relu')(merge_net)
    outputs = Dense(vocab_size, activation='softmax', name='caption_output')(merge_net)

    # Define model
    model = Model(
        inputs=[feature_input, topic_input, caption_input],
        outputs=outputs
    )
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
