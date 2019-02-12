""" Filter MSCOCO 2017 dataset and create a simplified version.
    The simplified version is stored in a pickle file.
"""


import os
import json
import pickle
import argparse


def get_categories(categories_file):
    """ Group categories by image
    """
    # map each category id to its name
    id_to_category = {}
    for category in categories_file['categories']:
        id_to_category[category['id']] = category['name']

    image_categories = {}
    for category in categories_file['annotations']:
        if category['image_id'] not in image_categories:
            image_categories[category['image_id']] = []
        if id_to_category[category['category_id']] not in image_categories[category['image_id']]:
            image_categories[category['image_id']].append(id_to_category[category['category_id']])
    return image_categories


def get_captions(captions):
    """ Group captions by image """
    image_captions = {}
    for caption in captions:
        img_id = caption['image_id']
        if not img_id in image_captions:
            image_captions[img_id] = []
        parsed_caption = caption['caption'].strip()
        parsed_caption = ''.join(parsed_caption.split('\n'))  # remove '\n' from the end of the caption
        image_captions[img_id].append(parsed_caption)
    return image_captions


def get_filename(images):
    """ Get filename of each image """
    image_file = {}
    for image in images:
        image_file[image['id']] = os.path.join(image['coco_url'].split('/')[-2], image['file_name'])
    return image_file


def group_supercategories(categories):
    """ Group supercategories by categories
    """
    cat_to_super = {}
    for category in categories:
        cat_to_super[category['name']] = category['supercategory']
    return cat_to_super


def get_supercategories(image_categories, cat_to_super):
    """ Group supercategories by image """
    image_supercategories = {}
    for image in image_categories:
        image_supercategories[image] = list(set([cat_to_super[x] for x in image_categories[image]]))
    return image_supercategories


def map_category_id(category_map):
    """ Assign an ID to each category """
    category_id = {}
    id_category = {}
    counter = 0
    for category in category_map:
        category_id[category['name']] = counter
        id_category[counter] = category['name']
        counter += 1
    return category_id, id_category


def parse_data(image_categories, image_supercategories, image_captions, image_file):
    images_data = {}
    for image in image_categories:
        images_data[image] = {
            'file_name': image_file[image],
            'supercategories': image_supercategories[image],
            'categories': image_categories[image],
            'captions': image_captions[image]
        }
    return images_data


def save_data(images_data, category_id, id_category, root_dir):
    """ Save parsed dataset """
    print('\nSaving raw dataset...')
    
    coco_raw = {
        'images_data': images_data,
        'category_id': category_id,
        'id_category': id_category
    }

    out_path = '{}/coco_raw.pickle'.format(root_dir)
    pickle_out = open(out_path, 'wb')
    pickle.dump(coco_raw, pickle_out)
    pickle_out.close()

    print('Done.')
    print('\n Data saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse MSCOCO dataset')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing the dataset folders'
    )
    args = parser.parse_args()

    
    # load annotations
    print('Loading instances and annotations...')
    captions_file = json.load(open('{}/annotations/captions_train2017.json'.format(root_dir), 'r'))
    categories_file = json.load(open('{}/annotations/instances_train2017.json'.format(root_dir), 'r'))
    print('Done.')

    # group categories by image
    image_categories = get_categories(categories_file)

    # group supercategories by image
    cat_to_super = group_supercategories(categories_file['categories'])
    image_supercategories = get_supercategories(image_categories, cat_to_super)

    # group captions by image
    image_captions = get_captions(captions_file['annotations'])

    # get filename of each image
    image_file = get_filename(captions_file['images'])

    # get complete dataset
    images_data = parse_data(image_categories, image_supercategories, image_captions, image_file)

    # assign each category an id.
    # we are not using the default ids given in the dataset because
    # the id ranges are not continuous.
    category_id, id_category = map_category_id(categories_file['categories'])

    # save parsed coco dataset
    save_data(images_data, category_id, id_category, root_dir)
