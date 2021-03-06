import os
import sys
import argparse
import pickle
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer

from dataset.process_texts import (
    mark_captions, flatten
)
from models.caption_model import create_model
from generator import batch_generator


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


def create_tokenizer(captions_marked):
    captions_flat = flatten(captions_marked)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_flat)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def calculate_steps_per_epoch(captions_list, batch_size):
    return int(len(captions_list) / batch_size)


def train(model, generator_train, generator_val, captions_train, captions_val, args):
    # define callbacks
    path_checkpoint = os.path.join(args.weights, 'cp-{epoch:02d}-v{val_loss:.2f}.hdf5')
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callback_tensorboard = TensorBoard(
        log_dir=os.path.join(args.weights, 'caption-logs'),
        histogram_freq=0,
        write_graph=True
    )
    callback_early_stop = EarlyStopping(monitor='val_loss', patience=args.early_stop, verbose=1)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay, patience=4, verbose=1, min_lr=args.min_lr)
    callbacks = [callback_checkpoint, callback_tensorboard, callback_early_stop, callback_reduce_lr]

    # train model
    try:
        model.fit_generator(
            generator=generator_train,
            steps_per_epoch=calculate_steps_per_epoch(captions_train, args.batch_size),
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=generator_val,
            validation_steps=calculate_steps_per_epoch(captions_val, args.batch_size)
        )
        print('\n\nModel training finished.')
    except KeyboardInterrupt:
        print('\n\nModel Training Interrupted.')
    
    return model


def save_model(model, weights_dir):
    """ Save the trained model with the best possible weights """

    # extract the filename of the weights with the lowest possible
    # validation loss value encountered during training
    weights_list = []
    for content in os.listdir(weights_dir):
        if content.startswith('cp-'):
            content_split = os.path.splitext(content)[0].split('-')
            loss = float(content_split[-1][1:])
            epoch = int(content_split[1])
            weights_list.append((content, loss, epoch))
    
    best_weights = os.path.join(
        weights_dir,
        sorted(weights_list, key=lambda x: (x[1], x[2]))[0][0]
    )

    # load the model with the best weights and save it to a file
    print('\n\nUsing weights file', best_weights, 'to save the model...')
    model.load_weights(best_weights)
    model.save(os.path.join(weights_dir, 'caption_model.hdf5'))
    print('\nModel saved to', os.path.join(weights_dir, 'caption_model.hdf5'))


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
    captions_train_marked = mark_captions(captions_train, mark_start, mark_end)  # training
    captions_val_marked = mark_captions(captions_val, mark_start, mark_end)  # validation
    tokenizer, vocab_size = create_tokenizer(captions_train_marked)

    # save the word_idx and idx_word dictionaries in a file
    # this will be required during evaluation
    word_idx_path = os.path.join(args.data, 'word_idx.pkl')
    idx_word_path = os.path.join(args.data, 'idx_word.pkl')
    with open(word_idx_path, mode='wb') as f:
        pickle.dump(tokenizer.word_index, f)
    with open(idx_word_path, mode='wb') as f:
        pickle.dump(tokenizer.index_word, f)

    num_classes = topic_transfer_values_train.shape[1]

    # training-dataset generator
    generator_train = batch_generator(
        topic_transfer_values_train,
        feature_transfer_values_train,
        captions_train_marked,
        tokenizer,
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
        tokenizer,
        len(captions_val),
        args.batch_size,
        args.max_tokens,
        vocab_size
    )

    # Create Model
    model = create_model(
        args.image_weights,
        args.state_size,
        args.dropout,
        tokenizer.word_index,
        args.glove,
        mark_start,
        mark_end,
        vocab_size,
        args.max_tokens
    )

    # train the model
    model = train(
        model,
        generator_train,
        generator_val,
        captions_train_marked,
        captions_val_marked,
        args
    )

    # save the model with the best weights
    save_model(model, args.weights)


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
        '--weights',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'),
        help='Directory in which to save the weights.'
    )
    parser.add_argument(
        '--glove',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'glove.6B.300d.txt'),
        help='Path to pre-trained GloVe vectors'
    )
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images per batch')
    parser.add_argument('--epochs', default=40, type=int, help='Epochs')
    parser.add_argument('--word_freq', default=5, type=int, help='Min frequency of words to consider for the vocabulary')
    parser.add_argument('--state_size', default=1024, type=int, help='State size of LSTM')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout Rate')
    parser.add_argument('--early_stop', default=12, type=int, help='Patience for early stopping callback')
    parser.add_argument('--lr_decay', default=0.2, type=float, help='Learning rate decay factor')
    parser.add_argument('--min_lr', default=0.00001, type=float, help='Lower bound on learning rate')
    parser.add_argument(
        '--image_weights',
        required=True,
        help='Path to weights of the topic model'
    )
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)
