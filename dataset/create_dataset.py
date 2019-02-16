import os
import random
import pickle
import argparse


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


def encode_categories(categories_list, category_id):
    categories_encoded = []
    for categories in categories_list:
        encode = [0] * len(category_id)
        for category in categories:
            encode[category_id[category]] = 1
        categories_encoded.append(encode)
    return categories_encoded


def load_coco(input_path, label, split_train=0.8, split_val=0.19):
    """ Load coco dataset """
    with open(input_path, 'rb') as file:
        coco_raw = pickle.load(file)
    images_data = coco_raw['images_data']
    category_id = coco_raw['category_id']
    id_category = coco_raw['id_category']
    
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
        train_data[1] = encode_categories(train_data[1], category_id)
        val_data[1] = encode_categories(val_data[1], category_id)
        test_data[1] = encode_categories(test_data[1], category_id)
    
    return train_data, val_data, test_data, category_id, id_category


def main(args):
    pass


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
