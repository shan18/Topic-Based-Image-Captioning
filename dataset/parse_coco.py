""" Filter MSCOCO 2017 dataset and create a simplified version.
    The simplified version is stored in a pickle file containing two dictionaries:
        - image_to_captions
        - image_to_file
"""


import os
import argparse
import pickle
from pycocotools.coco import COCO


def load_image_ids(data_dir, data_type):
    coco = COCO('{}/annotations/instances_{}.json'.format(data_dir, data_type))
    categories = coco.loadCats(coco.getCatIds())
    sports_categories = [
        category['id'] for category in categories if category['supercategory'] == 'sports'
    ]
    img_ids = set()
    for category in sports_categories:
        img_ids.update(coco.getImgIds(catIds=category))
    return list(img_ids)


def load_image_captions(coco, img_id):
    annotation_ids = coco.getAnnIds(imgIds=img_id)
    annotations = [annotation['caption'].strip() for annotation in coco.loadAnns(annotation_ids)]
    return annotations


def load_captions(coco, img_to_captions, img_ids):
    for img_id in img_ids:
        img_to_captions[img_id] = load_image_captions(coco, img_id)
    return img_to_captions


def load_filenames(coco, img_to_file, img_ids):
    for img in coco.loadImgs(img_ids):
        img_to_file[img['id']] = '/'.join(img['coco_url'].split('/')[-2:])
    return img_to_file


def parse_data(data_dir, data_type, img_to_captions, img_to_file):
    img_ids = load_image_ids(data_dir, data_type)
    coco_caps = COCO('{}/annotations/captions_{}.json'.format(data_dir, data_type))
    img_to_captions = load_captions(coco_caps, img_to_captions, img_ids)
    img_to_file = load_filenames(coco_caps, img_to_file, img_ids)
    return img_to_captions, img_to_file


def save_data(img_to_captions, img_to_file, data_dir):
    """ Save parsed dataset """
    print('\nSaving raw dataset...')
    
    coco_raw = {
        'image_to_captions': img_to_captions,
        'image_to_file': img_to_file
    }

    out_path = '{}/coco_raw.pickle'.format(data_dir)
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

    img_to_captions = {}
    img_to_file = {}
    
    img_to_captions, img_to_file = parse_data(args.root, 'train2017', img_to_captions, img_to_file)
    img_to_captions, img_to_file = parse_data(args.root, 'val2017', img_to_captions, img_to_file)

    save_data(img_to_captions, img_to_file, args.root)
