import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caption_category_model_train import load_data, mark_captions, create_tokenizer
from models.caption_category_model import create_model
from dataset.utils import print_progress_bar


def generate_predictions(topic_values, feature_values, caption_model, tokenizer, mark_start, mark_end, max_tokens):
    # Start with the initial start token
    predicted_caption = mark_start
    
    # Input for the caption model
    x_data = {
        'topic_input': np.expand_dims(topic_values, axis=0),
        'feature_input': np.expand_dims(feature_values, axis=0)
    }
    
    for i in range(max_tokens):
        sequence = tokenizer.texts_to_sequences([predicted_caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_tokens, padding='post')
        
        # predict next word
        x_data['caption_input'] = np.array(sequence)
        y_pred = caption_model.predict(x_data, verbose=0)
        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word[y_pred]
        
        # stop if we cannot map the word
        if word is None:
            break
            
        # append as input for generating the next word
        predicted_caption += ' ' + word
        
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    
    return ' '.join(predicted_caption.split()[1:-1])


def beam_search_predictions(
    topic_values, feature_values, caption_model, tokenizer, mark_start, mark_end, max_tokens, beam_width=5
):
    # Start with the initial start token
    start_word = [tokenizer.word_index[mark_start]]
    predictions = [[start_word, 1.0]]
    
    # Input for the caption model
    x_data = {
        'topic_input': np.expand_dims(topic_values, axis=0),
        'feature_input': np.expand_dims(feature_values, axis=0)
    }
    
    while len(predictions[0][0]) < max_tokens:
        temp = []
        
        for prediction in predictions:
            sequence = pad_sequences([prediction[0]], maxlen=max_tokens, padding='post')

            # predict next top words
            x_data['caption_input'] = sequence
            y_pred = caption_model.predict(x_data, verbose=0)
            word_predictions = np.argsort(y_pred[0])[-beam_width:]
            
            # create a new list to store the new sequences
            for word in word_predictions:
                caption = prediction[0] + [word]
                # probability = prediction[1] + math.log(y_pred[0][word])
                probability = prediction[1] * y_pred[0][word]
                temp.append([caption, probability])
        
        predictions = temp
        predictions = sorted(predictions, key=lambda x: x[1])  # sort according to the probabilities
        predictions = predictions[-beam_width:]  # Get the top words
    
    predictions = predictions[-1][0]
    half_captions = tokenizer.sequences_to_texts([predictions])[0].split()
        
    full_caption = []
    for word in half_captions[1:]:
        if word == mark_end:
            break
        full_caption.append(word)
    
    return ' '.join(full_caption)


def evaluate_model(
    topic_values, feature_values, caption_model, captions, tokenizer, mark_start, mark_end, max_tokens, mode, beam_width
):
    actual, predicted = [], []
    captions_length = len(captions)
    
    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, captions_length)
    
    for idx in range(captions_length):
        # generate description
        if mode == 'beam':
            y_pred = beam_search_predictions(
                topic_values[idx],
                feature_values[idx],
                caption_model,
                tokenizer,
                mark_start,
                mark_end,
                max_tokens,
                beam_width=beam_width
            )
        else:
            y_pred = generate_predictions(
                topic_values[idx],
                feature_values[idx],
                caption_model,
                tokenizer,
                mark_start,
                mark_end,
                max_tokens
            )
        
        # store actual and predicted
        references = [caption.split() for caption in captions[idx]]
        actual.append(references)
        predicted.append(y_pred.split())
        
        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, captions_length)

    print()
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def main(args):
    # Load pre-processed data
    _, _, captions_train = load_data(
        'train', args.data
    )
    topic_values, feature_values, captions_test = load_data(
        'test', args.data
    )

    # create tokenizer
    mark_start = 'startseq'
    mark_end = 'endseq'
    captions_train_marked = mark_captions(captions_train, mark_start, mark_end)
    tokenizer, vocab_size = create_tokenizer(captions_train_marked)

    # Get data size
    max_tokens = 16

    # Create Model
    model = create_model(
        topic_values.shape[1:],
        feature_values.shape[1:],
        tokenizer.word_index,
        args.glove,
        mark_start,
        mark_end,
        vocab_size,
        max_tokens
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
    evaluate_model(
        topic_values,
        feature_values,
        model,
        captions_test,
        tokenizer,
        mark_start,
        mark_end,
        max_tokens,
        args.mode,
        args.beam_width
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))
        ), 'dataset', 'processed_caption_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--glove',
        default=os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ), 'dataset', 'glove.6B.300d.txt'),
        help='Path to pre-trained GloVe vectors'
    )
    parser.add_argument(
        '--model_weights',
        required=True,
        help='Path to weights of the captioning model'
    )
    parser.add_argument(
        '--mode', choices=['beam', 'argmax'],
        required=True,
        help='Mode for predicting captions'
    )
    parser.add_argument(
        '--beam_width', type=int, default=5,
        help='Width of Beam Search'
    )
    args = parser.parse_args()

    main(args)