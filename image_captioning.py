import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset.utils import load_coco, load_image, print_progress_bar
from image_model.topic_layers import load_topic_model, load_feature_model


def load_raw_data(coco_raw_path):
    train_data, val_data, test_data, category_id, id_category = load_coco(
        coco_raw_path, 'captions', 
    )
    num_classes = len(id_category)
    return train_data, val_data, test_data, category_id, id_category, num_classes


def load_pre_trained_model(weights_path, num_classes):
    topic_model = load_topic_model(num_classes, weights_path)
    feature_model = load_feature_model()
    return topic_model, feature_model


def process_images(topic_model, feature_model, data_dir, filenames, batch_size):
    """
    Process all the given files in the given data_dir using the
    pre-trained topic-model as well as the feature-model and return
    their transfer-values.
    
    The images are processed in batches to save
    memory and improve efficiency.
    """
    
    # Number of images to process.
    num_images = len(filenames)
    
    # Get the expected input size of the pre-trained network
    img_size = K.int_shape(topic_model.input)[1:3]

    # Pre-allocate input-batch-array for images.
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)

    # Pre-allocate output-array for transfer-values.
    topic_transfer_values = np.zeros(
        shape=(num_images,) + K.int_shape(topic_model.output)[1:],
        dtype=np.int
    )
    feature_transfer_values = np.zeros(
        shape=(num_images, K.int_shape(feature_model.output)[1]),
        dtype=np.float32
    )

    # Initialize index into the filenames.
    start_index = 0

    # Process batches of image-files.
    while start_index < num_images:
        # Print the percentage-progress.
        print_progress_bar(start_index, num_images)

        # End-index for this batch.
        end_index = start_index + batch_size

        # Ensure end-index is within bounds.
        if end_index > num_images:
            end_index = num_images

        # The last batch may have a different batch-size.
        current_batch_size = end_index - start_index

        # Load all the images in the batch.
        for i, filename in enumerate(filenames[start_index:end_index]):
            # Path for the image-file.
            path = os.path.join(data_dir, filename)
            
            # Load and resize the image.
            # This returns the image as a numpy-array.
            img = load_image(path, size=img_size)

            # Save the image for later use.
            image_batch[i] = img

        # Use the pre-trained models to process the image.
        topic_predictions = topic_model.predict(
            image_batch[0:current_batch_size]
        )
        topic_transfer_values_batch = (topic_predictions > 0.5).astype('int')
        
        feature_transfer_values_batch = feature_model.predict(
            image_batch[0:current_batch_size]
        )

        # Save the transfer-values in the pre-allocated arrays.
        topic_transfer_values[start_index:end_index] = topic_transfer_values_batch[0:current_batch_size]
        feature_transfer_values[start_index:end_index] = feature_transfer_values_batch[0:current_batch_size]

        # Increase the index for the next loop-iteration.
        start_index = end_index

    # Print newline.
    print()

    return topic_transfer_values, feature_transfer_values


def process_data(topic_model, feature_model, data_dir, data_type, filenames, captions, batch_size):
    print('Processing {0} images in training-set ...'.format(len(filenames)))

    # Path for the cache-file.
    cache_path_dir = os.path.join(data_dir, 'processed_caption_data')
    topic_cache_path = os.path.join(
        cache_path_dir, 'topic_transfer_values_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        cache_path_dir, 'feature_transfer_values_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        cache_path_dir, 'captions_{}.pkl'.format(data_type)
    )
    
    # Check if directory to store processed data exists
    if not os.path.exists(cache_path_dir):
        print('Directory created:', cache_path_dir)
        os.mkdir(cache_path_dir)

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    if os.path.exists(topic_cache_path) and os.path.exists(feature_cache_path) and os.path.exists(captions_cache_path):
        with open(topic_cache_path, mode='rb') as file:
            topic_obj = pickle.load(file)
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
        with open(captions_cache_path, mode='rb') as file:
            captions = pickle.load(file)
        print("Data loaded from cache-file.")
    else:
        topic_obj, feature_obj = process_images(
            topic_model, feature_model, data_dir, filenames, batch_size
        )
        with open(topic_cache_path, mode='wb') as file:
            pickle.dump(topic_obj, file)
        with open(feature_cache_path, mode='wb') as file:
            pickle.dump(feature_obj, file)
        with open(captions_cache_path, mode='wb') as file:
            pickle.dump(captions, file)
        print("Data saved to cache-file.")

    return topic_obj, feature_obj, captions


def mark_captions(captions_list, mark_start, mark_end):
    """ Mark all the captions with the start and the end marker """
    captions_marked = [
        [mark_start + caption + mark_end for caption in captions] for captions in captions_list
    ]
    
    return captions_marked


def flatten(captions_list):
    """ Flatten all the captions into a single list """
    caption_list = [caption
                    for caption_list in captions_list
                    for caption in caption_list]
    
    return caption_list


def create_tokenizer(captions_marked):
    captions_flat = flatten(captions_marked)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_flat)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def create_sequences(tokenizer, max_length, topic_transfer_value, feature_transfer_value, caption, vocab_size):
    """ Create sequences of topic_values, feature_values, input sequence and output sequence for an image """
    topic_values, feature_values = [], []
    input_captions, output_captions = [], []
    integer_sequence = tokenizer.texts_to_sequences([caption])[0]  # encode the sequence
    
    for idx in range(1, len(integer_sequence)):
        in_seq, out_seq = integer_sequence[:idx], integer_sequence[idx]  # split into input and output pair
        in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post', truncating='post')[0]  # pad input sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]  # encode output sequence
        
        # store
        topic_values.append(topic_transfer_value)
        feature_values.append(feature_transfer_value)
        input_captions.append(in_seq)
        output_captions.append(out_seq)
        
    return topic_values, feature_values, input_captions, output_captions


def batch_generator(topic_transfer_values, feature_transfer_values, captions_list, tokenizer, num_images, batch_size, max_length, vocab_size):
    """
    Generator function for creating random batches of training-data.
    
    It selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the dataset.
        indices = np.random.randint(num_images, size=batch_size)
        
        # For a batch of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random
        topic_values, feature_values = [], []
        input_captions, output_captions = [], []
        for idx in indices:
            topic_value, feature_value, input_caption, output_caption = create_sequences(
                tokenizer,
                max_length,
                topic_transfer_values[idx],
                feature_transfer_values[idx],
                np.random.choice(captions_list[idx]),
                vocab_size
            )
            topic_values.extend(topic_value)
            feature_values.extend(feature_value)
            input_captions.extend(input_caption)
            output_captions.extend(output_caption)

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = {
            'caption_input': np.array(input_captions),
            'topic_input': np.array(topic_values),
            'feature_input': np.array(feature_values)
        }

        # Dict for the output-data.
        y_data = {
            'caption_output': np.array(output_captions)
        }
        
        yield (x_data, y_data)


def read_glove_vecs(glove_file):
    print('Creating word to vec map...')
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)
    print('Done!')
    return word_to_vec_map


def create_embedding_layer(word_to_index, word_to_vec_map, num_words):
    """ Create a Keras Embedding() layer and load in pre-trained GloVe 100-dimensional vectors
        @params:
        :word_to_index -- dictionary containing the each word mapped to its index
        :word_to_vec_map -- dictionary mapping words to their GloVe vector representation
        :num_words -- number of words in the vocabulary
        
        @return:
        :decoder_embedding -- pretrained layer Keras instance
    """
    
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


def create_model(topic_model, feature_model, tokenizer, word_to_vec_map, vocab_size):
    state_size = 256

    # Encode Images
    feature_input = Input(
        shape=K.int_shape(feature_model.output)[1:], name='feature_input'
    )
    image_model_output = Dense(state_size, activation='relu', name='image_model_output')(feature_input)

    # Encode Captions

    # Define layers
    topic_input = Input(
        shape=K.int_shape(topic_model.output)[1:], name='topic_input'
    )
    caption_input = Input(shape=(None,), name='caption_input')
    caption_embedding = create_embedding_layer(tokenizer.word_index, word_to_vec_map, vocab_size)
    caption_lstm = LSTM(state_size, name='caption_lstm')

    # connect layers
    topic_input_reshaped = Reshape(target_shape=(K.int_shape(topic_input)[1:] + (1,)))(topic_input)
    _, initial_state_h0, initial_state_c0 = LSTM(
        state_size, return_state=True, name='topic_lstm'
    )(topic_input_reshaped)
    topic_lstm_states = [initial_state_h0, initial_state_c0]
    net = caption_input  # Start the decoder-network with its input-layer
    net = caption_embedding(net)  # Connect the embedding-layer
    caption_model_output = caption_lstm(net, initial_state=topic_lstm_states) # Connect the caption LSTM layer
    
    # merge models
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


def calculate_steps_per_epoch(captions_list, batch_size):
    # Number of captions for each image in the dataset
    num_captions = [len(captions) for captions in captions_list]

    # Total number of captions in the training-set
    total_num_captions = np.sum(num_captions)

    # Approximate number of batches required per epoch,
    # if we want to process each caption and image pair once per epoch
    steps_per_epoch = int(total_num_captions / batch_size)
    return steps_per_epoch


def train(model, generator, num_images, captions_list, args):
    # define callbacks
    path_checkpoint = 'weights/checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True
    )
    callback_tensorboard = TensorBoard(
        log_dir='./weights/logs/',
        histogram_freq=0,
        write_graph=False
    )
    callbacks = [callback_checkpoint, callback_tensorboard]

    # train model
    model.fit_generator(
        generator=generator,
        steps_per_epoch=calculate_steps_per_epoch(captions_list, args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks
    )


def main(args):
    train_data, val_data, test_data, category_id, id_category, num_classes = load_raw_data(args.raw)
    train_images, train_captions = train_data  # Load training data
    val_images, val_captions = val_data  # Load validation data
    test_images, test_captions = test_data  # Load test data

    topic_model, feature_model = load_pre_trained_model(args.topic_weights, num_classes)

    process_batch_size = 64

    topic_transfer_values_train, feature_transfer_values_train, captions_train = process_data(
        topic_model, feature_model, 'dataset', 'train', train_images, train_captions, process_batch_size
    )
    print("topic shape:", topic_transfer_values_train.shape)
    print("feature shape:", feature_transfer_values_train.shape)

    topic_transfer_values_val, feature_transfer_values_val, captions_val = process_data(
        topic_model, feature_model, 'dataset', 'val', val_images, val_captions, process_batch_size
    )
    print("topic shape:", topic_transfer_values_val.shape)
    print("feature shape:", feature_transfer_values_val.shape)

    # process captions
    mark_start = 'startseq '
    mark_end = ' endseq'
    captions_train_marked = mark_captions(captions_train, mark_start, mark_end)
    tokenizer, vocab_size = create_tokenizer(captions_train_marked)

    # generator
    generator_train = batch_generator(
        topic_transfer_values_train,
        feature_transfer_values_train,
        captions_train_marked,
        tokenizer,
        len(train_images),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # embeddings
    word_to_vec_map = read_glove_vecs('{}/glove.6B.300d.txt'.format('dataset'))
    size = word_to_vec_map['unk'].shape
    word_to_vec_map[mark_start.strip()] = np.random.uniform(low=-1.0, high=1.0, size=size)
    word_to_vec_map[mark_end.strip()] = np.random.uniform(low=-1.0, high=1.0, size=size)

    # model
    model = create_model(topic_model, feature_model, tokenizer, word_to_vec_map, vocab_size)
    train(model, generator_train, len(train_images), captions_train, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_caption_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--raw',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument('--batch_size', default=6, type=int, help='Number of images per batch')
    parser.add_argument('--epochs', default=30, type=int, help='Epochs')
    parser.add_argument('--checkpoint', default='checkpoint', help='Filename to store model weights')
    parser.add_argument(
        '--topic_weights',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topic_extraction', 'weights', 'checkpoint.keras'),
        help='Path to weights of the topic model'
    )
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)
