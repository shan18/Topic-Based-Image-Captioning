import os
import random
import pickle
import argparse
import h5py
import numpy as np

from utils import load_image, print_progress_bar


def load_images_data(img_ids, images_data, label):
    filenames = []
    categories = []
    captions = []
    for img_id in img_ids:
        filenames.append(images_data[img_id]['file_name'])
        categories.append(images_data[img_id]['categories'])
        captions.append(images_data[img_id]['captions'])
    
    if label == 'categories':
        return (filenames, categories)
    return (filenames, captions)


def create_multi_label_categories_vector(categories_list, category_id):
    categories_encoded = []
    for categories in categories_list:
        encode = [0] * len(category_id)
        for category in categories:
            encode[category_id[category]] = 1
        categories_encoded.append(encode)
    return categories_encoded


def load_coco(input_path, label, split_train, split_val):
    """ Load coco dataset """
    with open(input_path, 'rb') as file:
        coco_raw = pickle.load(file)
    images_data = coco_raw['images_data']
    category_id = coco_raw['category_id']
    
    # split dataset
    img_ids = list(images_data.keys())
    split_idx_train = int(len(img_ids) * split_train)
    split_idx_val = int(len(img_ids) * split_val)
    random.shuffle(img_ids)
    img_ids_train = img_ids[:split_idx_train]
    img_ids_val = img_ids[split_idx_train:split_idx_train+split_idx_val]
    img_ids_test = img_ids[split_idx_train+split_idx_val:]
    
    # load dataset
    train_data = load_images_data(img_ids_train, images_data, label)  # training dataset
    val_data = load_images_data(img_ids_val, images_data, label)  # validation dataset
    test_data = load_images_data(img_ids_test, images_data, label)  # test dataset

    if label == 'categories':
        # encode categories
        train_data[1] = create_multi_label_categories_vector(train_data[1], category_id)
        val_data[1] = create_multi_label_categories_vector(val_data[1], category_id)
        test_data[1] = create_multi_label_categories_vector(test_data[1], category_id)
    
    return train_data, val_data, test_data, category_id


def encode_images_list(filenames, root, image_size, grayscale):
    """ Store images in a numpy array """

    images = []
    dataset_size = len(filenames)

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size)
    
    for path in filenames:
        img_array = load_image(
            os.path.join(root, path),
            size=(image_size, image_size),
            grayscale=grayscale
        )
        images.append(img_array)

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, dataset_size)
        
    return np.array(images)


def encode_images(filenames, root, image_size, grayscale, dataset_type):
    print('Processing {} images in {}-set ...'.format(len(filenames), dataset_type))

    # Path for the cache-file.
    cache_path = os.path.join(root, '{}_images.h5'.format(dataset_type))

    # If the cache-file already exists then skip,
    # otherwise process all images and save their encodings
    # to the cache-file so it can be reloaded quickly.
    if os.path.exists(cache_path):
        print("Cache-file: " + cache_path + "already exists.")
    else:
        # The cache-file does not exist.
        images = encode_images_list(filenames, root, image_size, grayscale)

        # Save the data to a cache-file.
        h5f = h5py.File(cache_path, 'w')
        h5f.create_dataset('images', data=images)
        h5f.close()

        print("- Data saved to cache-file: " + cache_path)


def encode_categories(labels, label_type, root, dataset_type):
    print('Processing {} image labels in {}-set ...'.format(len(labels), dataset_type))

    # Path for the cache-file.
    cache_path = os.path.join(root, '{}_{}.h5'.format(dataset_type, label_type))

    # If the cache-file exists.
    if os.path.exists(cache_path):
        print("Cache-file: " + cache_path + "already exists.")
    else:
        # Save the data to a cache-file.
        h5f = h5py.File(cache_path, 'w')
        h5f.create_dataset('labels', data=labels)
        h5f.close()

        print("Data saved to cache-file: " + cache_path)


def main(args):
    train_data, val_data, test_data, category_id = load_coco(
        args.raw, args.label, args.split_train, args.split_val
    )
    filenames_train, labels_train = train_data
    filenames_val, labels_val = val_data
    filenames_test, labels_test = test_data

    # load and store images
    encode_images(filenames_train, args.root, args.image_size, args.grayscale, 'train')
    encode_images(filenames_val, args.root, args.image_size, args.grayscale, 'val')
    encode_images(filenames_test, args.root, args.image_size, args.grayscale, 'test')

    # load and store categories
    encode_categories(labels_train, args.label, args.root, 'train')
    encode_categories(labels_val, args.label, args.root, 'val')
    encode_categories(labels_test, args.label, args.root, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing the dataset folders'
    )
    parser.add_argument(
        '--raw', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument('--label', choices=['categories', 'captions'], help='Type of label vector to create')
    parser.add_argument('--split_train', default=0.8, help='Training data split')
    parser.add_argument('--split_val', default=0.19, help='Validation data split')
    parser.add_argument('--image_size', default=224, type=int, help='Image size to use in dataset')
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
