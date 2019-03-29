from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Add

from models.vgg19 import load_vgg19
from models.category_model import load_category_model
from models.encoder import image_encoder, caption_encoder


def load_pre_trained_image_model(weights_path, num_classes):
    topic_model = load_category_model(num_classes, weights_path)
    feature_model = load_vgg19()
    print('Done.\n')
    return topic_model, feature_model


def create_model(image_model_weights, num_topics, word_index, glove_file, mark_start, mark_end, vocab_size, max_tokens=16):
    state_size = 256

    # Load pre-trained image model
    topic_model, feature_model = load_pre_trained_image_model(image_model_weights, num_topics)

    # Encode Images
    feature_input, image_model_output = image_encoder(feature_model, state_size)

    # Encode Captions
    topic_input_shape = K.int_shape(topic_model.output)[1:]
    topic_input, caption_input, caption_model_output = caption_encoder(
        topic_input_shape, word_index, glove_file, mark_start, mark_end, state_size, vocab_size, max_tokens
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

