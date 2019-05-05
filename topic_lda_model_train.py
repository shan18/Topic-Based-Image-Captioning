import os
import argparse
import pickle
import sys
import h5py
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from models.topic_model import create_topic_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    feature_cache_path = os.path.join(
        data_dir, 'features_{}.pkl'.format(data_type)
    )
    topics_cache_path = os.path.join(
        data_dir, 'lda_topics_{}.pkl'.format(data_type)
    )

    if os.path.exists(topics_cache_path):
        with open(topics_cache_path, mode='rb') as file:
            topics = pickle.load(file)
    if os.path.exists(feature_cache_path):
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
    else:
        sys.exit('processed {} data does not exist.'.format(data_type))

    print('{} data loaded from cache-file.'.format(data_type))
    return feature_obj, topics


def train_model(model, train_data, val_data, args):
    train_images, train_categories = train_data

    # set weights directory and checkpoint path
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    path_checkpoint = 'weights/lda-topic-weights-{epoch:02d}-{val_loss:.2f}.hdf5'

    # set model callbacks
    callback_tensorboard = TensorBoard(
        log_dir=os.path.join(weights_dir, 'topic-category-logs'),
        histogram_freq=0,
        write_graph=True
    )
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay, patience=4, verbose=1, min_lr=args.min_lr)
    # early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks = [callback_tensorboard, callback_checkpoint, callback_reduce_lr]

    try:
        model.fit(
            x=train_images,
            y=train_categories,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=val_data,
            shuffle='batch'
        )
        print('\n\nModel training finished.')
    except KeyboardInterrupt:
        print('Stopped.')


def main(args):
    # Load pre-processed data
    features_train, topics_train = load_data(
        'train', args.data
    )
    features_val, topics_val = load_data(
        'val', args.data
    )
    # topics_train = np.array(topics_train)
    # topics_val = np.array(topics_val)
    print('\nFeatures shape:', features_train.shape)
    print('Topics shape:', topics_train.shape)

    # Create model
    model = create_topic_model(features_train.shape[1:], topics_train.shape[1])
    print(model.summary())

    # Train model
    train_model(model, (features_train, topics_train), (features_val, topics_val), args)


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
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=40, type=int, help='Epochs')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='Lower bound on learning rate')
    args = parser.parse_args()

    main(args)
