import os
import argparse
import pickle
import numpy as np
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_coco, load_image, print_progress_bar
from models.vgg19 import load_vgg19


def process_images(feature_model, filenames, args):
    """
    Process all the given files in the given data_dir using the
    pre-trained feature-model as well as the feature-model and return
    their transfer-values.
    
    The images are processed in batches to save memory and improve efficiency.
    """
    
    num_images = len(filenames)
    img_size = K.int_shape(feature_model.input)[1:3]    # Expected input size of the pre-trained network

    # Pre-allocate input-batch-array for images.
    shape = (args.batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)

    # Pre-allocate output-array for transfer-values.
    feature_transfer_values = np.zeros(
        shape=(num_images, K.int_shape(feature_model.output)[1]),
        dtype=np.float32
    )

    start_index = 0
    print_progress_bar(start_index, num_images)  # Initial call to print 0% progress

    while start_index < num_images:
        end_index = start_index + args.batch_size
        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index

        # Load all the images in the batch.
        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(args.root, filename)
            img = load_image(path, size=img_size, grayscale=args.grayscale)
            image_batch[i] = img

        # Use the pre-trained models to process the image.
        feature_transfer_values_batch = feature_model.predict(
            image_batch[0:current_batch_size]
        )

        # Save the transfer-values in the pre-allocated array.
        feature_transfer_values[start_index:end_index] = feature_transfer_values_batch[0:current_batch_size]

        start_index = end_index
        print_progress_bar(start_index, num_images)  # Update Progress Bar

    print()
    return feature_transfer_values


def process_data(feature_model, data_type, filenames, captions, args):
    print('Processing {0} images in {1}-set ...'.format(len(filenames), data_type))

    # Path for the cache-file.
    cache_path_dir = os.path.join(args.root, 'processed_lda_data')
    feature_cache_path = os.path.join(
        cache_path_dir, 'feature_transfer_values_{}.pkl'.format(data_type)
    )
    images_cache_path = os.path.join(
        cache_path_dir, 'images_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        cache_path_dir, 'captions_{}.pkl'.format(data_type)
    )
    
    # Check if directory to store processed data exists
    if not os.path.exists(cache_path_dir):
        print('Directory created:', cache_path_dir)
        os.mkdir(cache_path_dir)

    # Process all images and save their transfer-values
    feature_obj = process_images(
        feature_model, filenames, args
    )
    with open(feature_cache_path, mode='wb') as file:
        pickle.dump(feature_obj, file)
    with open(images_cache_path, mode='wb') as file:
        pickle.dump(filenames, file)
    with open(captions_cache_path, mode='wb') as file:
        pickle.dump(captions, file)
    print('{} data saved to cache-file.'.format(data_type))


def main(args):
    train_data, val_data, test_data, _, _ = load_coco(
        args.raw, 'captions', args.split_train, args.split_val
    )
    train_images, train_captions = train_data  # Load training data
    val_images, val_captions = val_data  # Load validation data
    test_images, test_captions = test_data  # Load test data
    
    # Load pre-trained image models
    feature_model = load_vgg19()

    # Generate and save dataset
    process_data(  # training data
        feature_model, 'train', train_images, train_captions, args
    )
    process_data(  # validation data
        feature_model, 'val', val_images, val_captions, args
    )
    process_data(  # test data
        feature_model, 'test', test_images, test_captions, args
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for caption model')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing the dataset folders'
    )
    parser.add_argument(
        '--raw', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='Batch size for the pre-trained model to make predictions'
    )
    parser.add_argument('--split_train', default=0.8, help='Training data split')
    parser.add_argument('--split_val', default=0.1, help='Validation data split')
    parser.add_argument('--grayscale', action='store_true', help='Images will be stored in grayscale')
    args = parser.parse_args()

    if args.split_train <= 0 or args.split_train >= 1:
        print('Value of split_train should be between 0 and 1')
    elif args.split_val <= 0 or args.split_val >= 1:
        print('Value of split_val should be between 0 and 1')
    elif args.split_train <= args.split_val:
        print('split_train should be greater than split_val')
    elif args.split_train + args.split_val >= 1:
        print('Please enter a valid split')
    else:
        main(args)

