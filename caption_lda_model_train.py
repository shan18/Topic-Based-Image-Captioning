import os
import sys
import argparse
import pickle
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.caption_lda_model import create_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    feature_cache_path = os.path.join(
        data_dir, 'feature_transfer_values_{}.pkl'.format(data_type)
    )
    topics_cache_path = os.path.join(
        data_dir, 'topics_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        data_dir, 'captions_{}.pkl'.format(data_type)
    )

    feature_path_exists = os.path.exists(feature_cache_path)
    topic_path_exists = os.path.exists(topics_cache_path)
    caption_path_exists = os.path.exists(captions_cache_path)
    if feature_path_exists and topic_path_exists and caption_path_exists:
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
        with open(topics_cache_path, mode='rb') as file:
            topics = pickle.load(file)
        with open(captions_cache_path, mode='rb') as file:
            captions = pickle.load(file)
    else:
        sys.exit('processed {} data does not exist.'.format(data_type))

    print('{} data loaded from cache-file.'.format(data_type))
    return feature_obj, topics, captions


def mark_captions(captions_list, mark_start, mark_end):
    """ Mark all the captions with the start and the end marker """
    captions_marked = [
        [' '.join([mark_start, caption, mark_end]) for caption in captions] for captions in captions_list
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


def batch_generator(
    topic_transfer_values, feature_transfer_values, captions_list, tokenizer, num_images, batch_size, max_length, vocab_size
):
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


def calculate_steps_per_epoch(captions_list, batch_size):
    # Number of captions for each image
    num_captions = [len(captions) for captions in captions_list]
    
    # Total number of captions
    total_num_captions = np.sum(num_captions)
    
    return int(total_num_captions / batch_size)


def train(model, generator_train, generator_val, captions_train, captions_val, args):
    # define callbacks
    path_checkpoint = 'weights/caption-lda.{epoch:02d}-{val_loss:.2f}.hdf5'
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callback_tensorboard = TensorBoard(
        log_dir='./weights/caption-lda-logs/',
        histogram_freq=0,
        write_graph=True
    )
    callback_early_stop = EarlyStopping(monitor='val_loss', patience=args.early_stop, verbose=1)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay, patience=4, verbose=1, min_lr=args.min_lr)
    callbacks = [callback_checkpoint, callback_tensorboard, callback_early_stop, callback_reduce_lr]

    # train model
    model.fit_generator(
        generator=generator_train,
        steps_per_epoch=calculate_steps_per_epoch(captions_train, args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=generator_val,
        validation_steps=calculate_steps_per_epoch(captions_val, args.batch_size)
    )

    print('\n\nModel training finished.')


def main(args):
    # Load pre-processed data
    features_train, topics_train, captions_train = load_data(
        'train', args.data
    )
    features_val, topics_val, captions_val = load_data(
        'val', args.data
    )
    print('\nFeatures shape:', features_train.shape[1:])
    print('Topics shape:', topics_train.shape[1:])

    # process captions
    mark_start = 'startseq'
    mark_end = 'endseq'
    captions_train_marked = mark_captions(captions_train, mark_start, mark_end)  # training
    captions_val_marked = mark_captions(captions_val, mark_start, mark_end)  # validation
    tokenizer, vocab_size = create_tokenizer(captions_train_marked)

    # training-dataset generator
    generator_train = batch_generator(
        topics_train,
        features_train,
        captions_train_marked,
        tokenizer,
        len(captions_train),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # validation-dataset generator
    generator_val = batch_generator(
        topics_val,
        features_val,
        captions_val_marked,
        tokenizer,
        len(captions_val),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # Create Model
    model = create_model(
        topics_train.shape[1:],
        features_train.shape[1:],
        tokenizer.word_index,
        args.glove,
        mark_start,
        mark_end,
        vocab_size,
        args.max_tokens
    )

    # train the model
    train(
        model,
        generator_train,
        generator_val,
        captions_train_marked,
        captions_val_marked,
        args
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_lda_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--glove',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'glove.6B.300d.txt'),
        help='Path to pre-trained GloVe vectors'
    )
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    parser.add_argument('--epochs', default=30, type=int, help='Epochs')
    parser.add_argument('--early_stop', default=12, type=int, help='Patience for early stopping callback')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='Lower bound on learning rate')
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)

