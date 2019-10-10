import os
import argparse
import pickle
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import load_image
from models.utils import load_pre_trained_image_model
from evaluation.predictions import generate_predictions


def load_data(data_dir):
    """ Load word index dictionary """
    word_idx_cache_path = os.path.join(
        data_dir, 'word_idx.pkl'
    )
    idx_word_cache_path = os.path.join(
        data_dir, 'idx_word.pkl'
    )

    word_idx_path_exists = os.path.exists(word_idx_cache_path)
    idx_word_path_exists = os.path.exists(idx_word_cache_path)
    if idx_word_path_exists and word_idx_path_exists:
        with open(word_idx_cache_path, mode='rb') as file:
            word_idx = pickle.load(file)
        with open(idx_word_cache_path, mode='rb') as file:
            idx_word = pickle.load(file)
        print("Dictionary loaded.")
    else:
        sys.exit('File containing the dictionary does not exist.')

    return word_idx, idx_word


def process_image(img_path, img_size):
    # Convert image to vector
    img = load_image(img_path, size=img_size, grayscale=False)

    # Create image batch
    shape = (1,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float32)
    image_batch[0] = img

    return image_batch


def get_model_input(topic_model_path, img_path):
    # Load pre-trained image models
    topic_model, feature_model = load_pre_trained_image_model(topic_model_path)

    # Load image batch
    image_batch = process_image(
        img_path, K.int_shape(feature_model.input)[1:3]
    )

    # Create input data
    feature_values = feature_model.predict(image_batch)
    topic_values = topic_model.predict(feature_values)

    return topic_values[0], feature_values[0]


def main(args):
    # Get input data
    topic_values, feature_values =  get_model_input(args.topic_model, args.image)

    # Load caption model
    caption_model = load_model(args.caption_model)

    # Load dictionaries
    word_idx, idx_word = load_data(
        args.data
    )

    # Get prediction
    predicted_caption = generate_predictions(
        topic_values,
        feature_values,
        caption_model,
        word_idx,
        idx_word,
        args.max_tokens
    )
    print('\nPrediction:', predicted_caption)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset', 'processed_data'
        ),
        help='Directory containing the word_index dictionary'
    )
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument(
        '--topic_model',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'topic_model.hdf5'),
        help='Path to the trained topic model'
    )
    parser.add_argument(
        '--caption_model',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'caption_model.hdf5'),
        help='Path to the trained caption model'
    )
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print('Image does not exist')
    elif not os.path.exists(args.caption_model):
        print('Dictionary does not exist')
    elif not os.path.exists(args.topic_model):
        print('Topic model does not exist')
    elif not os.path.exists(args.caption_model):
        print('Caption model does not exist')
    else:
        main(args)
