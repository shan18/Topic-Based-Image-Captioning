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

from dataset.utils import load_coco
from dataset.process_texts import (
    mark_captions, clean_captions, caption_to_sequence, build_vocabulary_with_frequency_threshold
)
from models.caption_model import create_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    topic_cache_path = os.path.join(
        data_dir, 'lda_topics_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        data_dir, 'features_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        data_dir, 'captions_{}.pkl'.format(data_type)
    )

    topic_path_exists = os.path.exists(topic_cache_path)
    feature_path_exists = os.path.exists(feature_cache_path)
    caption_path_exists = os.path.exists(captions_cache_path)
    if topic_path_exists and feature_path_exists and caption_path_exists:
        with open(topic_cache_path, mode='rb') as file:
            topic_obj = pickle.load(file)
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
        with open(captions_cache_path, mode='rb') as file:
            captions = pickle.load(file)
        print("Data loaded from cache-file.")
    else:
        sys.exit('File containing the processed data does not exist.')

    return np.array(topic_obj), feature_obj, captions


def process_captions(captions_list, mark_start, mark_end, freq_threshold):
    captions_list_marked = mark_captions(captions_list, mark_start, mark_end)
    captions_list_marked = clean_captions(captions_list_marked)
    vocab, word_idx, _ = build_vocabulary_with_frequency_threshold(captions_list_marked, freq_threshold)
    return captions_list_marked, word_idx, len(vocab) + 1


def create_sequences(word_idx, max_length, topic_transfer_value, feature_transfer_value, caption, vocab_size):
    """ Create sequences of topic_values, feature_values, input sequence and output sequence for an image """
    topic_values, feature_values = [], []
    input_captions, output_captions = [], []
    integer_sequence = caption_to_sequence(caption, word_idx)  # encode the sequence
    
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
    topic_transfer_values, feature_transfer_values, captions_list, word_idx, num_images, batch_size, max_length, vocab_size
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
                word_idx,
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
    path_checkpoint = 'weights/cp-{epoch:02d}-{val_loss:.2f}.hdf5'
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callback_tensorboard = TensorBoard(
        log_dir='./weights/caption-logs/',
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
    topic_transfer_values_train, feature_transfer_values_train, captions_train = load_data(
        'train', args.data
    )
    topic_transfer_values_val, feature_transfer_values_val, captions_val = load_data(
        'val', args.data
    )
    print("topic shape:", topic_transfer_values_train.shape)
    print("feature shape:", feature_transfer_values_train.shape)

    # process captions
    mark_start = 'startseq'
    mark_end = 'endseq'
    captions_train_marked, word_idx, vocab_size = process_captions(  # training
        captions_train, mark_start, mark_end, args.word_freq
    )
    captions_val_marked = mark_captions(captions_val, mark_start, mark_end)  # validation
    captions_val_marked = clean_captions(captions_val_marked)

    num_classes = topic_transfer_values_train.shape[1]

    # training-dataset generator
    generator_train = batch_generator(
        topic_transfer_values_train,
        feature_transfer_values_train,
        captions_train_marked,
        word_idx,
        len(captions_train),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # validation-dataset generator
    generator_val = batch_generator(
        topic_transfer_values_val,
        feature_transfer_values_val,
        captions_val_marked,
        word_idx,
        len(captions_val),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # Create Model
    model = create_model(
        args.image_weights,
        feature_transfer_values_train.shape[1:],
        num_classes,
        args.state_size,
        args.dropout,
        word_idx,
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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--raw',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--glove',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'glove.6B.300d.txt'),
        help='Path to pre-trained GloVe vectors'
    )
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    parser.add_argument('--epochs', default=30, type=int, help='Epochs')
    parser.add_argument('--word_freq', default=5, type=int, help='Min frequency of words to consider for the vocabulary')
    parser.add_argument('--state_size', default=1024, type=int, help='State size of LSTM')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout Rate')
    parser.add_argument('--early_stop', default=12, type=int, help='Patience for early stopping callback')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='Lower bound on learning rate')
    parser.add_argument(
        '--image_weights',
        required=True,
        help='Path to weights of the topic model'
    )
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)
