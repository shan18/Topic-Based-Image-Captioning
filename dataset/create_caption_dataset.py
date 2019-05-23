import os
import argparse
import pickle
import h5py
import numpy as np
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_coco, load_image, print_progress_bar
from create_topic_dataset import load_split_data
from models.inception_v3 import load_inception_v3
from models.topic_model import load_topic_model


def load_input_shape(data_type, data_dir):
    # Path for the cache-file.
    feature_cache_path = os.path.join(
        data_dir, 'inception_features_{}.h5'.format(data_type)
    )

    if os.path.exists(feature_cache_path):
        feature_file = h5py.File(feature_cache_path, 'r')
        feature_obj = feature_file['feature_values']
    else:
        sys.exit('processed {} data does not exist.'.format(data_type))
    
    input_shape = feature_obj.shape[1:]
    feature_file.close()
    return input_shape


def load_pre_trained_model(input_shape, output_dim, weights_path):
    print('Loading pre-trained models...')
    topic_model = load_topic_model(input_shape, output_dim, weights_path)
    feature_model = load_inception_v3()
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
    img_size = K.int_shape(feature_model.input)[1:3]  # Expected input size of the pre-trained network

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
            img = load_image(path, size=img_size, grayscale=False)
            image_batch[i] = img

        # Use the pre-trained models to process the image
        feature_transfer_values_batch = feature_model.predict(
            image_batch[0:current_batch_size]
        )
        topic_transfer_values_batch = topic_model.predict(
            feature_transfer_values_batch
        )

        # Save the transfer-values in the pre-allocated arrays
        topic_transfer_values[start_index:end_index] = topic_transfer_values_batch[0:current_batch_size]
        feature_transfer_values[start_index:end_index] = feature_transfer_values_batch[0:current_batch_size]

        start_index = end_index
        print_progress_bar(start_index, num_images)  # Update Progress Bar

    print()
    return topic_transfer_values, feature_transfer_values


def process_data(topic_model, feature_model, img_ids, filenames, categories, captions, data_type, args):
    print('Processing {0} images in {1}-set ...'.format(len(filenames), data_type))

    # Path for the cache-file.
    cache_path_dir = os.path.join(args.root, 'processed_data')
    topic_cache_path = os.path.join(
        cache_path_dir, 'topics_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        cache_path_dir, 'features_{}.pkl'.format(data_type)
    )
    images_id_cache_path = os.path.join(
        cache_path_dir, 'images_id_{}.pkl'.format(data_type)
    )
    images_cache_path = os.path.join(
        cache_path_dir, 'images_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        cache_path_dir, 'captions_{}.pkl'.format(data_type)
    )
    categories_cache_path = os.path.join(
        cache_path_dir, 'categories_{}.pkl'.format(data_type)
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
    with open(images_id_cache_path, mode='wb') as file:
        pickle.dump(img_ids, file)
    with open(images_cache_path, mode='wb') as file:
        pickle.dump(filenames, file)
    with open(categories_cache_path, mode='wb') as file:
        pickle.dump(categories, file)
    with open(captions_cache_path, mode='wb') as file:
        pickle.dump(captions, file)
    print("Data saved to cache-file.")


def main(args):
    train_data, val_data, test_data = load_split_data(
        args.raw, args.split
    )
    train_img_ids, train_images, train_categories, train_captions = train_data
    val_img_ids, val_images, val_categories, val_captions = val_data
    test_img_ids, test_images, test_categories, test_captions = test_data

    if args.num_classes is None:
        num_classes = len(train_categories[0])
    else:
        num_classes = args.num_classes
    print('\nNum Topics:', num_classes)
    
    # Load pre-trained image models
    topic_model, feature_model = load_pre_trained_model(
        load_input_shape('train', args.data), num_classes, args.image_weights
    )

    # Generate and save dataset
    process_data(  # training data
        topic_model, feature_model, train_img_ids, train_images, train_categories, train_captions, 'train', args
    )
    process_data(  # validation data
        topic_model, feature_model, val_img_ids, val_images, val_categories, val_captions, 'val', args
    )
    process_data(  # test data
        topic_model, feature_model, test_img_ids, test_images, test_categories, test_captions, 'test', args
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
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--image_weights', required=True,
        help='Path to weights of the topic model'
    )
    parser.add_argument(
        '--batch_size', default=256, type=int,
        help='Batch size for the pre-trained model to make predictions'
    )
    parser.add_argument(
        '--num_classes', default=None, type=int,
        help='Number of classes for the model'
    )
    parser.add_argument('--split', default=5000, help='Number of images for validation and test set')
    args = parser.parse_args()

    main(args)
