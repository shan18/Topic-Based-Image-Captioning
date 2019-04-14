import os
import sys
import argparse
import pickle
import h5py
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from models.topic_model import create_topic_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    feature_cache_path = os.path.join(
        data_dir, 'vgg_features_{}.h5'.format(data_type)
    )
    topics_cache_path = os.path.join(
        data_dir, 'topics_{}.pkl'.format(data_type)
    )

    if os.path.exists(topics_cache_path):
        with open(topics_cache_path, mode='rb') as file:
            topics = pickle.load(file)
    if os.path.exists(feature_cache_path):
        feature_file = h5py.File(feature_cache_path, 'r')
        feature_obj = feature_file['feature_values']
    else:
        sys.exit('processed {} data does not exist.'.format(data_type))

    print('{} data loaded from cache-file.'.format(data_type))
    return feature_file, feature_obj, topics


def train(model, train_data, val_data, args):
    # dataset
    features_train, topics_train = train_data

    # define callbacks
    path_checkpoint = 'weights/topic-weights-{epoch:02d}-{val_loss:.2f}.hdf5'
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callback_tensorboard = TensorBoard(
        log_dir='./weights/topic-logs/',
        histogram_freq=0,
        write_graph=True
    )
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay, patience=2, verbose=1, min_lr=args.min_lr)
    callbacks = [callback_checkpoint, callback_tensorboard, callback_reduce_lr]

    # train model
    model.fit(
        x=features_train,
        y=topics_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_data,
        shuffle='batch'
    )

    print('\n\nModel training finished.')


def main(args):
    # Load pre-processed data
    feature_file_train, features_train, topics_train = load_data(
        'train', args.data
    )
    feature_file_val, features_val, topics_val = load_data(
        'val', args.data
    )
    features_val_arr = np.array(features_val)
    print('\nFeatures shape:', features_train.shape)
    print('Topics shape:', topics_train.shape)

    # Create model
    model = create_topic_model(features_train.shape[1:], topics_train.shape[1])
    print(model.summary())

    # Train model
    train(model, (features_train, topics_train), (features_val_arr, topics_val), args)

    # Close the dataset file
    feature_file_train.close()
    feature_file_val.close()


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
    parser.add_argument('--batch_size', default=16384, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=15, type=int, help='Epochs')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--min_lr', default=0.00001, type=float, help='Lower bound on learning rate')
    args = parser.parse_args()

    main(args)

