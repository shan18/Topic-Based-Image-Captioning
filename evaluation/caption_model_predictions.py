import os
import argparse
import json
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caption_model_train import mark_captions, create_tokenizer
from models.caption_model import create_model
from dataset.utils import print_progress_bar
from predictions import generate_predictions


def load_data(data_type, data_dir):
    # Path for the cache-file.
    topic_cache_path = os.path.join(
        data_dir, 'categories_{}.pkl'.format(data_type)
    )
    feature_cache_path = os.path.join(
        data_dir, 'features_{}.pkl'.format(data_type)
    )
    captions_cache_path = os.path.join(
        data_dir, 'captions_{}.pkl'.format(data_type)
    )
    img_ids_cache_path = os.path.join(
        data_dir, 'images_id_{}.pkl'.format(data_type)
    )

    topic_path_exists = os.path.exists(topic_cache_path)
    feature_path_exists = os.path.exists(feature_cache_path)
    caption_path_exists = os.path.exists(captions_cache_path)
    img_ids_path_exists = os.path.exists(img_ids_cache_path)
    if topic_path_exists and feature_path_exists and caption_path_exists and img_ids_path_exists:
        with open(topic_cache_path, mode='rb') as file:
            topic_obj = pickle.load(file)
        with open(feature_cache_path, mode='rb') as file:
            feature_obj = pickle.load(file)
        with open(captions_cache_path, mode='rb') as file:
            captions = pickle.load(file)
        with open(img_ids_cache_path, mode='rb') as file:
            img_ids = pickle.load(file)
        print("Data loaded from cache-file.")
    else:
        sys.exit('File containing the processed data does not exist.')

    return np.array(topic_obj), feature_obj, captions, img_ids


def store_predictions(
    topic_values, feature_values, caption_model, img_ids, tokenizer, mark_start, mark_end, max_tokens
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
            tokenizer,
            mark_start,
            mark_end,
            max_tokens
        )
        captions.append({'image_id': img_ids[idx], 'caption': y_pred})
        
        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, num_images)

    print()
    with open('captions_results.json', 'w') as f:
        json.dump(captions, f)


def main(args):
    # Load pre-processed data
    _, _, captions_train, _ = load_data(
        'train', args.data
    )
    topic_values, feature_values, _, img_ids = load_data(
        'test', args.data
    )

    # create tokenizer
    mark_start = 'startseq'
    mark_end = 'endseq'
    captions_train_marked = mark_captions(captions_train, mark_start, mark_end)
    tokenizer, vocab_size = create_tokenizer(captions_train_marked)

    # Create Model
    model = create_model(
        args.image_weights,
        feature_values.shape[1:],
        topic_values.shape[1],
        args.state_size,
        args.dropout,
        tokenizer.word_index,
        args.glove,
        mark_start,
        mark_end,
        vocab_size,
        args.max_tokens
    )

    # Load weights
    try:
        model.load_weights(args.model_weights)
        print('Weights loaded.')
    except Exception as e:
        print('Error loading the weights.')
        print(e)
        sys.exit(1)

    # Evaluate
    store_predictions(
        topic_values,
        feature_values,
        model,
        img_ids,
        tokenizer,
        mark_start,
        mark_end,
        args.max_tokens
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))
        ), 'dataset', 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--raw',
        default=os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ), 'dataset', 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--glove',
        default=os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ), 'dataset', 'glove.6B.300d.txt'),
        help='Path to pre-trained GloVe vectors'
    )
    parser.add_argument(
        '--image_weights', required=True,
        help='Path to weights of the topic model'
    )
    parser.add_argument(
        '--model_weights',
        required=True,
        help='Path to weights of the captioning model'
    )
    parser.add_argument('--state_size', default=1024, type=int, help='State size of LSTM')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout Rate')
    parser.add_argument('--max_tokens', default=16, type=int, help='Max length of the captions')
    args = parser.parse_args()

    main(args)
