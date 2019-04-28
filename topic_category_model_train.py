import os
import argparse
import pickle
import h5py
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.topic_category_model import create_category_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    feature_cache_path = os.path.join(
        data_dir, 'vgg_features_{}.h5'.format(data_type)
    )
    topics_cache_path = os.path.join(
        data_dir, 'categories_{}.pkl'.format(data_type)
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


def train_model(model, train_data, val_data, args):
    train_images, train_categories = train_data

    # set weights directory and checkpoint path
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    path_checkpoint = 'weights/topic-weights-{epoch:02d}-{val_loss:.2f}.hdf5'

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
    # early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks = [callback_tensorboard, callback_checkpoint]

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
    feature_file_train, features_train, topics_train = load_data(
        'train', args.data
    )
    feature_file_val, features_val, topics_val = load_data(
        'val', args.data
    )
    features_val_arr = np.array(features_val)
    print('\nFeatures shape:', features_train.shape)
    print('Topics shape:', topics_train.shape)

    # Load mapping
    with open(args.raw, 'rb') as file:
        coco_raw = pickle.load(file)
    id_category = coco_raw['id_category']

    # Create model
    model = create_category_model(topics_train.shape[1])
    print(model.summary())

    # Train model
    train_model(model, (features_train, topics_train), (features_val_arr, topics_val), args)

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
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs')
    args = parser.parse_args()

    main(args)