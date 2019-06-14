import os
import argparse
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import print_progress_bar
from predictions import generate_predictions


def load_data(data_type, data_dir):
    # Path for the cache-file.
    topic_cache_path = os.path.join(
        data_dir, 'lda_topics_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        data_dir, 'features_{}.pkl'.format(data_type)
    )
    img_ids_cache_path = os.path.join(
        data_dir, 'images_id_{}.pkl'.format(data_type)
    )
    word_idx_cache_path = os.path.join(
        data_dir, 'word_idx.pkl'.format(data_type)
    )
    idx_word_cache_path = os.path.join(
        data_dir, 'idx_word.pkl'.format(data_type)
    )

    topic_path_exists = os.path.exists(topic_cache_path)
    feature_path_exists = os.path.exists(feature_cache_path)
    img_ids_path_exists = os.path.exists(img_ids_cache_path)
    word_idx_path_exists = os.path.exists(word_idx_cache_path)
    idx_word_path_exists = os.path.exists(idx_word_cache_path)
    if topic_path_exists and feature_path_exists and idx_word_path_exists and img_ids_path_exists and word_idx_path_exists:
        with open(topic_cache_path, mode='rb') as file:
            topic_obj = pickle.load(file)
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
        with open(img_ids_cache_path, mode='rb') as file:
            img_ids = pickle.load(file)
        with open(word_idx_cache_path, mode='rb') as file:
            word_idx = pickle.load(file)
        with open(idx_word_cache_path, mode='rb') as file:
            idx_word = pickle.load(file)
        print("Data loaded from cache-file.")
    else:
        sys.exit('File containing the processed data does not exist.')

    return np.array(topic_obj), feature_obj, img_ids, word_idx, idx_word


def store_predictions(
    topic_values, feature_values, caption_model, img_ids, word_idx, idx_word, max_tokens
):
    captions = []
    num_images = len(img_ids)
    
    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, num_images)
    
    for idx in range(num_images):
        # generate description
        y_pred = generate_predictions(
            topic_values[idx],
            feature_values[idx],
            caption_model,
            word_idx,
            idx_word,
            max_tokens
        )
        captions.append({'image_id': img_ids[idx], 'caption': y_pred})
        
        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, num_images)

    print()
    with open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model_predictions.json'
    ), 'w') as f:
        json.dump(captions, f)


def main(args):
    # Load pre-processed test data
    topic_values, feature_values, img_ids, word_idx, idx_word = load_data(
        'test', args.data
    )

    # Load Model
    try:
        model = load_model(args.model)
        print('Model loaded.')
    except Exception as e:
        print('Error trying to load the model.')
        print(e)
        sys.exit(1)

    # Evaluate
    store_predictions(
        topic_values,
        feature_values,
        model,
        img_ids,
        word_idx,
        idx_word,
        args.max_tokens
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset', 'processed_data'
        ),
        help='Directory containing the processed dataset and word_index dictionary'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to the saved model'
    )
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)
