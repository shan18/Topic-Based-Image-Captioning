import os
import argparse
import pickle
import numpy as np
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_coco, load_image, print_progress_bar
from models.vgg19 import load_vgg19
from models.topic_category_model import load_category_model


def load_pre_trained_model(weights_path, num_classes):
    print('Loading pre-trained models...')
    topic_model = load_category_model(num_classes, weights_path)
    feature_model = load_vgg19()
    print('Done.\n')
    return topic_model, feature_model


def process_images(topic_model, feature_model, filenames, args):
    """
    Process all the given files in the given root path using the
    pre-trained topic-model as well as the feature-model and return
    their transfer-values.
    
    The images are processed in batches to save memory and improve efficiency.
    """
    
    num_images = len(filenames)
    img_size = K.int_shape(topic_model.input)[1:3]  # Expected input size of the pre-trained network

    # Pre-allocate input-batch-array for images
    shape = (args.batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)

    # Pre-allocate output-array for transfer-values.
    topic_transfer_values = np.zeros(
        shape=(num_images,) + K.int_shape(topic_model.output)[1:],
        dtype=np.float32
    )
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

        # Use the pre-trained models to process the image
        topic_transfer_values_batch = topic_model.predict(
            image_batch[0:current_batch_size]
        )
        feature_transfer_values_batch = feature_model.predict(
            image_batch[0:current_batch_size]
        )

        # Save the transfer-values in the pre-allocated arrays
        topic_transfer_values[start_index:end_index] = topic_transfer_values_batch[0:current_batch_size]
        feature_transfer_values[start_index:end_index] = feature_transfer_values_batch[0:current_batch_size]

        start_index = end_index
        print_progress_bar(start_index, num_images)  # Update Progress Bar

    print()
    return topic_transfer_values, feature_transfer_values


def process_data(topic_model, feature_model, filenames, data_type, args):
    print('Processing {0} images in {1}-set ...'.format(len(filenames), data_type))

    # Path for the cache-file.
    cache_path_dir = os.path.join(args.root, 'processed_data')
    topic_cache_path = os.path.join(
        cache_path_dir, 'topic_values_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        cache_path_dir, 'feature_values_{}.pkl'.format(data_type)
    )
    
    # Check if directory to store processed data exists
    if not os.path.exists(cache_path_dir):
        os.mkdir(cache_path_dir)
        print('Directory created:', cache_path_dir)

    # Process all images and save their transfer-values
    topic_obj, feature_obj = process_images(
        topic_model, feature_model, filenames, args
    )
    with open(topic_cache_path, mode='wb') as file:
        pickle.dump(topic_obj, file)
    with open(feature_cache_path, mode='wb') as file:
        pickle.dump(feature_obj, file)
    print("Data saved to cache-file.")


def main(args):
    train_data, val_data, test_data, category_id, _ = load_coco(
        args.raw, split
    )
    train_img_ids, train_images, train_categories, train_captions = train_data  # Load training data
    val_img_ids, val_images, val_categories, val_captions = val_data  # Load validation data
    test_img_ids, test_images, test_categories, test_captions = test_data  # Load test data

    num_classes = len(category_id)
    
    # Load pre-trained image models
    topic_model, feature_model = load_pre_trained_model(args.image_weights, num_classes)

    # Generate and save dataset
    process_data(  # training data
        topic_model, feature_model, train_images, 'train', args
    )
    process_data(  # validation data
        topic_model, feature_model, val_images, 'val', args
    )
    process_data(  # test data
        topic_model, feature_model, test_images, 'test', args
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
        '--image_weights', required=True,
        help='Path to weights of the topic model'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='Batch size for the pre-trained model to make predictions'
    )
    parser.add_argument('--split', default=5000, help='Number of images for validation and test set')
    parser.add_argument('--grayscale', action='store_true', help='Images will be stored in grayscale')
    args = parser.parse_args()

    main(args)
