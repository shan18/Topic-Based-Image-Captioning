from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add, LSTM, BatchNormalization, Dropout, Reshape

from models.vgg19 import load_vgg19_flatten
from models.encoder import image_encoder, caption_encoder


def create_model(
    topic_shape, feature_shape, state_size, word_index, glove_file, mark_start, mark_end, vocab_size, max_tokens=16
):
    # Encode Images
    feature_input, image_model_output = image_encoder(feature_shape, state_size)

    # Encode Captions
    topic_input, caption_input, caption_model_output = caption_encoder(
        topic_shape, word_index, glove_file, mark_start, mark_end, state_size, vocab_size, max_tokens
    )
    
    # merge encoders and create the decoder
    merge_net = Add()([image_model_output, caption_model_output])
    merge_net = Dense(state_size, activation='relu')(merge_net)
    # merge_net = Reshape(target_shape=(K.int_shape(merge_net)[1:] + (1,)))(merge_net)
    # merge_net = LSTM(state_size, name='merge_lstm')(merge_net)
    # merge_net = Dropout(0.5)(merge_net)
    # merge_net = BatchNormalization(name='merge_batch_normalize')(merge_net)
    outputs = Dense(vocab_size, activation='softmax', name='caption_output')(merge_net)

    # Define model
    model = Model(
        inputs=[feature_input, topic_input, caption_input],
        outputs=outputs
    )
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
