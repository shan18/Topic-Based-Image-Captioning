import os
import argparse
import h5py
import pickle
import numpy as np
import random
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_coco, load_image, print_progress_bar
from models.inception_v3 import load_inception_v3


def create_multi_label_categories_vector(categories_list, category_id):
    categories_encoded = []
    for categories in categories_list:
        encode = [0] * len(category_id)
        for category in categories:
            encode[category_id[category]] = 1
        categories_encoded.append(encode)
    return categories_encoded


def load_split_data(input_path, split):
    """ Load coco dataset """
    train_data, val_data, test_data, category_id, _ = load_coco(
        input_path, split
    )
    train_img_ids, train_images, train_categories, train_captions = train_data  # Load training data
    val_img_ids, val_images, val_categories, val_captions = val_data  # Load validation data
    test_img_ids, test_images, test_categories, test_captions = test_data  # Load test data

    train_categories = create_multi_label_categories_vector(train_categories, category_id)
    val_categories = create_multi_label_categories_vector(val_categories, category_id)
    test_categories = create_multi_label_categories_vector(test_categories, category_id)
    
    return (train_img_ids, train_images, train_categories, train_captions), (val_img_ids, val_images, val_categories, val_captions), (test_img_ids, test_images, test_categories, test_captions)


def process_images(model, filenames, data_dir, save_file, batch_size):
    num_images = len(filenames)
    img_size = K.int_shape(model.input)[1:3]    # Expected input size of the pre-trained network

    # Pre-allocate input-batch-array for images.
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)

    start_index = 0
    print_progress_bar(start_index, num_images)  # Initial call to print 0% progress
    
    with h5py.File(save_file, 'w') as data_file:
        # Pre-allocate output-array for transfer-values.
        feature_transfer_values = data_file.create_dataset(
            'feature_values', shape=(num_images, K.int_shape(model.output)[1]), dtype=np.float32, chunks=True
        )
        while start_index < num_images:
            end_index = start_index + batch_size
            if end_index > num_images:
                end_index = num_images
            current_batch_size = end_index - start_index

            # Load all the images in the batch.
            for i, filename in enumerate(filenames[start_index:end_index]):
                path = os.path.join(data_dir, filename)
                img = load_image(path, size=img_size)
                image_batch[i] = img

            # Use the pre-trained models to process the image.
            feature_transfer_values_batch = model.predict(
                image_batch[0:current_batch_size]
            )

            # Save the transfer-values in the pre-allocated array.
            feature_transfer_values[start_index:end_index] = feature_transfer_values_batch[0:current_batch_size]

            start_index = end_index
            print_progress_bar(start_index, num_images)  # Update Progress Bar

    print()


def process_data(model, data_type, filenames, categories, captions, save_path, data_dir, batch_size):
    print('Processing {0} images in {1}-set ...'.format(len(filenames), data_type))

    # Path for the cache-file.
    cache_path_dir = save_path
    feature_cache_path = os.path.join(
        cache_path_dir, 'inception_features_{}.h5'.format(data_type)
    )
    categories_cache_path = os.path.join(
        cache_path_dir, 'categories_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        cache_path_dir, 'captions_{}.pkl'.format(data_type)
    )
    
    # Check if directory to store processed data exists
    if not os.path.exists(cache_path_dir):
        os.mkdir(cache_path_dir)
        print('Directory created:', cache_path_dir)

    # Process all images and save their transfer-values
    process_images(
        model, filenames, data_dir, feature_cache_path, batch_size
    )
    with open(categories_cache_path, mode='wb') as file:
        pickle.dump(categories, file)
    with open(captions_cache_path, mode='wb') as file:
        pickle.dump(captions, file)
    print('{} data saved to cache-file.'.format(data_type))


def main(args):
    train_data, val_data, test_data = load_split_data(
        args.raw, args.split
    )
    _, train_images, train_categories, train_captions = train_data
    _, val_images, val_categories, val_captions = val_data
    _, test_images, test_categories, test_captions = test_data

    print('\nDataset sizes:')
    print('Training:', len(train_images))
    print('Validation:', len(val_images))
    print('Test:', len(test_images))

    model = load_inception_v3()

    # Generate and save dataset
    process_data(  # training data
        model, 'train', train_images, train_categories, train_captions, args.save, args.root, args.batch_size
    )
    process_data(  # validation data
        model, 'val', val_images, val_categories, val_captions, args.save, args.root, args.batch_size
    )
    process_data(  # test data
        model, 'test', test_images, test_categories, test_captions, args.save, args.root, args.batch_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for image model')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing the dataset folders'
    )
    parser.add_argument(
        '--raw', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--save',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--split', default=5000, help='Number of images for validation and test set')
    args = parser.parse_args()

    main(args)
