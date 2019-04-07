import os
import argparse
import h5py
import pickle
import numpy as np
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import print_progress_bar
from models.topic_model import load_feature_model


def load_data(data_type, data_dir):
    # Path for the cache-file.
    vgg_feature_cache_path = os.path.join(
        data_dir, 'vgg_features_{}.h5'.format(data_type)
    )
    topics_cache_path = os.path.join(
        data_dir, 'topics_{}.pkl'.format(data_type)
    )

    if os.path.exists(topics_cache_path):
        with open(topics_cache_path, mode='rb') as file:
            topics = pickle.load(file)
    if os.path.exists(vgg_feature_cache_path):
        vgg_feature_file = h5py.File(vgg_feature_cache_path, 'r')
        vgg_feature_obj = vgg_feature_file['feature_values']
    else:
        sys.exit('processed {} data does not exist.'.format(data_type))

    print('{} data loaded from cache-file.'.format(data_type))
    return vgg_feature_file, vgg_feature_obj, topics


def process_data(feature_model, vgg_features, save_dir, data_type, batch_size):
    save_file = os.path.join(save_dir, 'features_{}.h5'.format(data_type))
    num_images = vgg_features.shape[0]

    start_index = 0
    print_progress_bar(start_index, num_images)  # Initial call to print 0% progress
    
    with h5py.File(save_file, 'w') as data_file:
        # Pre-allocate output-array for transfer-values.
        feature_values = data_file.create_dataset(
            'feature_values', shape=(num_images, K.int_shape(feature_model.output)[1]), dtype=np.float32, chunks=True
        )
        while start_index < num_images:
            end_index = start_index + batch_size
            if end_index > num_images:
                end_index = num_images
            current_batch_size = end_index - start_index

            # Use the pre-trained models to process the image.
            feature_values_batch = feature_model.predict(
                vgg_features[0:current_batch_size]
            )

            # Save the transfer-values in the pre-allocated array.
            feature_values[start_index:end_index] = feature_values_batch[0:current_batch_size]

            start_index = end_index
            print_progress_bar(start_index, num_images)  # Update Progress Bar

    print()


def main(args):
    # Load pre-processed data
    vgg_feature_file_train, vgg_features_train, topics_train = load_data(
        'train', args.data
    )
    vgg_feature_file_val, vgg_features_val, _ = load_data(
        'val', args.data
    )
    vgg_feature_file_test, vgg_features_test, _ = load_data(
        'test', args.data
    )
    print('\nFeatures shape:', vgg_features_train.shape)
    
    # Load pre-trained feature model
    feature_model = load_feature_model(vgg_features_train.shape[1:], topics_train.shape[1], args.weights)

    # Generate and save dataset
    process_data(  # training data
        feature_model, vgg_features_train, args.data, 'train', args.batch_size
    )
    process_data(  # validation data
        feature_model, vgg_features_val, args.data, 'val', args.batch_size
    )
    process_data(  # test data
        feature_model, vgg_features_test, args.data, 'test', args.batch_size
    )

    # Close the dataset file
    vgg_feature_file_train.close()
    vgg_feature_file_val.close()
    vgg_feature_file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for caption model')
    parser.add_argument(
        '--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--weights', required=True,
        help='Topic model weights'
    )
    parser.add_argument(
        '--batch_size', default=512, type=int,
        help='Batch size for the pre-trained model to make predictions'
    )
    args = parser.parse_args()

    main(args)
