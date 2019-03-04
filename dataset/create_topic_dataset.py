import os
import argparse
import h5py
import numpy as np

from utils import load_coco, load_image, print_progress_bar


def create_multi_label_categories_vector(categories_list, category_id):
    categories_encoded = []
    for categories in categories_list:
        encode = [0] * len(category_id)
        for category in categories:
            encode[category_id[category]] = 1
        categories_encoded.append(encode)
    return categories_encoded


def load_split_data(input_path, split_train, split_val):
    """ Load coco dataset """
    train_data, val_data, test_data, category_id, id_category = load_coco(input_path, 'categories', split_train, split_val)
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data

    train_labels = create_multi_label_categories_vector(train_labels, category_id)
    val_labels = create_multi_label_categories_vector(val_labels, category_id)
    test_labels = create_multi_label_categories_vector(test_labels, category_id)
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), category_id


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
    cache_path = os.path.join(root, 'processed_topic_data/{}_images.h5'.format(dataset_type))

    # If the cache-file already exists then skip,
    # otherwise process all images and save their encodings
    # to the cache-file so it can be reloaded quickly.
    if os.path.exists(cache_path):
        print("Cache-file: " + cache_path + " already exists.")
    else:
        # The cache-file does not exist.
        images = encode_images_list(filenames, root, image_size, grayscale)

        # Save the data to a cache-file.
        h5f = h5py.File(cache_path, 'w')
        h5f.create_dataset('images', data=images)
        h5f.close()

        print("- Data saved to cache-file: " + cache_path)


def encode_categories(labels, root, dataset_type):
    print('Processing {} image labels in {}-set ...'.format(len(labels), dataset_type))

    # Path for the cache-file.
    cache_path = os.path.join(root, 'processed_topic_data/{}_categories.h5'.format(dataset_type))

    # If the cache-file exists.
    if os.path.exists(cache_path):
        print("Cache-file: " + cache_path + " already exists.")
    else:
        # Save the data to a cache-file.
        h5f = h5py.File(cache_path, 'w')
        h5f.create_dataset('labels', data=labels)
        h5f.close()

        print("Data saved to cache-file: " + cache_path)


def main(args):
    train_data, val_data, test_data, category_id = load_split_data(
        args.raw, args.split_train, args.split_val
    )
    filenames_train, labels_train = train_data
    filenames_val, labels_val = val_data
    filenames_test, labels_test = test_data

    # check if path to save data exists
    save_path = os.path.join(args.root, 'processed_topic_data')
    if not os.path.exists(save_path):
        print('Directory created:', save_path)
        os.mkdir(save_path)

    # load and store images
    encode_images(filenames_train, args.root, args.image_size, args.grayscale, 'train')
    encode_images(filenames_val, args.root, args.image_size, args.grayscale, 'val')
    encode_images(filenames_test, args.root, args.image_size, args.grayscale, 'test')

    # load and store categories
    encode_categories(labels_train, args.root, 'train')
    encode_categories(labels_val, args.root, 'val')
    encode_categories(labels_test, args.root, 'test')


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
    parser.add_argument('--split_train', default=0.8, help='Training data split')
    parser.add_argument('--split_val', default=0.1, help='Validation data split')
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
