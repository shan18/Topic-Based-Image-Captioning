import os
import sys
import argparse
import pickle
import lda
import numpy as np

from dataset.process_texts import remove_stopwords, apply_stemming, create_text_matrix


def load_data(data_type, data_dir):
    # Path for the cache-file.
    captions_cache_path = os.path.join(
        data_dir, 'captions_{}.pkl'.format(data_type)
    )

    if os.path.exists(captions_cache_path):
        with open(captions_cache_path, mode='rb') as file:
            captions = pickle.load(file)
        print('{} data loaded from cache-file.'.format(data_type))
        print('{} data size: {}'.format(data_type, len(captions)))
    else:
        sys.exit('File containing the processed data does not exist.')

    return captions


def create_and_train(lda_data, num_topics, iterations):
    model = lda.LDA(n_topics=num_topics, n_iter=iterations, random_state=1)
    model.fit(lda_data)
    return model


def save_topics(topics, data_dir, data_type):
    topics_path = os.path.join(data_dir, 'topics_{}.pkl'.format(data_type))
    with open(topics_path, mode='wb') as file:
        pickle.dump(topics, file)
    print('Data Saved.')


def create_lda_data(captions, num_words):
    captions = remove_stopwords(captions)
    captions = apply_stemming(captions)
    lda_data = create_text_matrix(captions, num_words)
    return lda_data


def main(args):
    # Load pre-processed data
    captions_train = load_data(
        'train', args.data
    )
    captions_val = load_data(
        'val', args.data
    )

    # Create data
    lda_data_train = create_lda_data(captions_train, args.num_words)
    lda_data_val = create_lda_data(captions_val, args.num_words)

    # Create and train model
    model = create_and_train(lda_data_train, args.topics, args.iterations)

    # Get topics
    train_topic = model.doc_topic_
    val_topic = model.transform(lda_data_val)

    # Save
    save_topics(train_topic, args.data, 'train')
    save_topics(val_topic, args.data, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument('--num_words', default=5000, type=int, help='Number of words to keep in vocabulary')
    parser.add_argument('--topics', default=80, type=int, help='Number of topics in the data')
    parser.add_argument('--iterations', default=2000, type=int, help='Number of iterations for fitting the model')
    args = parser.parse_args()

    main(args)
