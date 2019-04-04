import os
import argparse
import pickle
import random
import numpy as np
from tensorflow.keras import backend as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_images_data, load_image, print_progress_bar
from models.vgg19 import load_vgg19


def load_coco(input_path, label, split):
    """ Load coco dataset """
    with open(input_path, 'rb') as file:
        coco_raw = pickle.load(file)
    images_data_train = coco_raw['images_data_train']
    images_data_val = coco_raw['images_data_val']
    category_id = coco_raw['category_id']
    id_category = coco_raw['id_category']
    
    # split dataset
    img_ids = list(images_data_train.keys())
    random.shuffle(img_ids)

    img_ids_val = list(images_data_val.keys())[:split]
    val_split_diff = split - len(img_ids_val)
    if val_split_diff > 0:
        for img_id in img_ids_train[:val_split_diff]:
            img_ids_val.append(img_id)
            images_data_val[img_id] = images_data_train[img_id]

    img_ids_test = img_ids[val_split_diff:split + val_split_diff]
    img_ids_train = img_ids[split + val_split_diff:]
    
    # load dataset
    train_images, train_labels = load_images_data(img_ids_train, images_data_train, label)  # training dataset
    val_images, val_labels = load_images_data(img_ids_val, images_data_val, label)  # validation dataset
    test_images, test_labels = load_images_data(img_ids_test, images_data_train, label)  # test dataset
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), category_id, id_category


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
    cache_path_dir = args.save
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
        args.raw, 'captions', args.split
    )
    train_images, train_captions = train_data  # Load training data
    val_images, val_captions = val_data  # Load validation data
    test_images, test_captions = test_data  # Load test data
    
    # Load pre-trained image models
    feature_model = load_vgg19()

    print('\nDataset sizes:')
    print('Training:', len(train_images))
    print('Validation:', len(val_images))
    print('Test:', len(test_images))

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
        '--raw', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_raw_all.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--save', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_all_data'),
        help='Directory to store processed dataset'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='Batch size for the pre-trained model to make predictions'
    )
    parser.add_argument('--split', default=5000, help='Number of images for validation and test set')
    parser.add_argument('--grayscale', action='store_true', help='Images will be stored in grayscale')
    args = parser.parse_args()

    main(args)

