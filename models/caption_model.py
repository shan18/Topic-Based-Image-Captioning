from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add

from models.encoder import create_image_encoder, create_caption_encoder
from models.utils import load_pre_trained_image_model


def create_model(
    image_model_weights, state_size, dropout,
    word_idx, glove_file, mark_start, mark_end, vocab_size, max_tokens=16
):
    # Load pre-trained image model
    topic_model, feature_model = load_pre_trained_image_model(image_model_weights)

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
