import os
import argparse
import h5py
import pickle
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


def encode_images_list(filenames, cache_path, root, image_size, grayscale):
    """ Store images in a numpy array """

    dataset_size = len(filenames)

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size)
    
    with h5py.File(cache_path, 'w') as data_file:
        images = data_file.create_dataset(
            'images', shape=(dataset_size, image_size, image_size, 3), dtype=np.float32, chunks=True
        )
        for idx, path in enumerate(filenames):
            img_array = load_image(
                os.path.join(root, path),
                size=(image_size, image_size),
                grayscale=grayscale
            )
            images[idx] = img_array

            # Update Progress Bar
            print_progress_bar_counter += 1
            print_progress_bar(print_progress_bar_counter, dataset_size)
            
        print()


def encode_images(filenames, root, image_size, grayscale, dataset_type):
    print('Processing {} images in {}-set ...'.format(len(filenames), dataset_type))

    # Path for the cache-file.
    cache_path = os.path.join(root, 'processed_data/{}_images.h5'.format(dataset_type))

    # If the cache-file already exists then skip,
    # otherwise process all images and save their encodings
    # to the cache-file so it can be reloaded quickly.
    if os.path.exists(cache_path):
        print("Cache-file: " + cache_path + " already exists.")
    else:
        # The cache-file does not exist.
        encode_images_list(filenames, cache_path, root, image_size, grayscale)

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
        h5f.create_dataset('categories', data=labels)
        h5f.close()

        print("Data saved to cache-file: " + cache_path)


def save_rest_data(img_ids, filenames, captions, data_type, root):
    # Path for the cache-file.
    cache_path_dir = os.path.join(root, 'processed_data')
    images_id_cache_path = os.path.join(
        cache_path_dir, 'images_id_{}.pkl'.format(data_type)
    )
    images_cache_path = os.path.join(
        cache_path_dir, 'filenames_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        cache_path_dir, 'captions_{}.pkl'.format(data_type)
    )

    with open(images_id_cache_path, mode='wb') as file:
        pickle.dump(img_ids, file)
    with open(images_cache_path, mode='wb') as file:
        pickle.dump(filenames, file)
    with open(captions_cache_path, mode='wb') as file:
        pickle.dump(captions, file)


def main(args):
    train_data, val_data, test_data = load_split_data(
        args.raw, args.split
    )
    train_img_ids, train_images, train_categories, train_captions = train_data
    val_img_ids, val_images, val_categories, val_captions = val_data
    test_img_ids, test_images, test_categories, test_captions = test_data

    # check if path to save data exists
    save_path = os.path.join(args.root, 'processed_data')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('Directory created:', save_path)

    # load and store images
    encode_images(train_images, args.root, args.image_size, args.grayscale, 'train')
    encode_images(val_images, args.root, args.image_size, args.grayscale, 'val')
    encode_images(test_images, args.root, args.image_size, args.grayscale, 'test')

    # load and store categories
    encode_categories(train_categories, args.root, 'train')
    encode_categories(val_categories, args.root, 'val')
    encode_categories(test_categories, args.root, 'test')

    save_rest_data(train_img_ids, train_images, train_captions, 'train', args.root)
    save_rest_data(val_img_ids, val_images, val_captions, 'val', args.root)
    save_rest_data(test_img_ids, test_images, test_captions, 'test', args.root)


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
    parser.add_argument('--split', default=5000, help='Number of images for validation and test set')
    parser.add_argument('--image_size', default=224, type=int, help='Image size to use in dataset')
    parser.add_argument('--grayscale', action='store_true', help='Images will be stored in grayscale')
    args = parser.parse_args()

    main(args)
