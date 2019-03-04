import cv2
import pickle
import random
import numpy as np


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


def load_coco(input_path, label, split_train=0.8, split_val=0.1):
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
    train_images, train_labels = load_images_data(img_ids_train, images_data, label)  # training dataset
    val_images, val_labels = load_images_data(img_ids_val, images_data, label)  # validation dataset
    test_images, test_labels = load_images_data(img_ids_test, images_data, label)  # test dataset
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), category_id, id_category


def load_image(path, size=None, grayscale=False):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """

    # Load the image using opencv
    if not grayscale:  # RGB format
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:  # grayscale format
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize image if desired.
    if not size is None:
        image = cv2.resize(image, size)

    # Convert image to numpy array and scale pixels so they fall between 0.0 and 1.0
    image = np.array(image) / 255.0

    # Add 1 extra dimension to grayscale images
    if (len(image.shape) == 2):
        image = np.expand_dims(image, axis=-1)

    return image


def print_progress_bar(iteration, total):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(50 * iteration // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    print('\rProgress: |%s| %s%% Complete' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
